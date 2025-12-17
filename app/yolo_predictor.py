import os
import yaml
import pandas as pd
import logging
from ultralytics import YOLO
from datetime import datetime
import ultralytics
import torch
from app.config_loader import load_config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def should_ignore_class(cls_id: int, class_names: list) -> bool:
    """
    Returns True if the predicted class should be ignored based on rules.
    Currently ignores any class whose name contains '700ml' or '750ml'.
    """
    name = str(class_names[cls_id]).lower() if cls_id < len(class_names) else ""
    ignore_keywords = ["700ml", "750ml"]
    return any(keyword in name for keyword in ignore_keywords)
def extract_brand(name: str):
    lname = name.lower()
    if "coke" in lname or "coca" in lname:
        return "coke"
    if "sprite" in lname:
        return "sprite"
    if "fanta" in lname:
        return "fanta"
    if "pepsi" in lname:
        return "pepsi"
    if "kinley" in lname:
        return "kinley"
    return None


def extract_size(name: str):
    lname = name.lower()
    for tok in ["2250", "1500", "1000", "750", "700", "600", "500", "250", "175"]:
        if tok in lname:
            return tok
    return None


def is_cap_class(class_name: str) -> bool:
    lname = class_name.lower()
    cap_keywords = [
        "cap",
        "caps"
    ]
    exclude_keywords = [
        "bottle",
        "pet",
        "glass",
        "can",
        "cooler",
        "visicooler",
        "shelf"
    ]

    if any(ex in lname for ex in exclude_keywords):
        return False

    return any(k in lname for k in cap_keywords)



def run_yolo_predictions(yaml_path, model_path, image_folder, csv_output_path, modelname, s3_bucket_name, s3_folder, conn, cur, s3_handler, image_paths, cyclecountid_override=None):
    """Run YOLO predictions and save results to CSV and database."""
    logger.info(f"Using ultralytics version: {ultralytics.__version__}")
    logger.info(f"Using torch version: {torch.__version__}")
    
    from app.db_handler import get_max_cyclecountid, clear_cyclecount_staging, insert_yolo_prediction
    try:
        # Use pipeline cyclecountid if provided
        if cyclecountid_override is not None:
            cyclecountid = cyclecountid_override
            logger.info(f"Using cyclecountid from pipeline: {cyclecountid}")
        else:
            cyclecountid = get_max_cyclecountid(cur) + 1
            logger.info(f"YOLO computed cyclecountid: {cyclecountid}")
        logger.info(f"Using cyclecountid: {cyclecountid}")
    except Exception as e:
        logger.error(f"Failed to retrieve cyclecountid, using default: {e}")
        cyclecountid = 0

    s3_cycle_folder = f"ModelResults/CycleCount_{cyclecountid}"

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        model = YOLO(model_path)
        class_names = model.names
        logger.info(f"Loaded class names from {yaml_path}: {class_names}")
    except Exception as e:
        logger.error(f"Failed to load YAML file {yaml_path}: {e}")
        raise

    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    save_dir = None
    predict_dir = "runs/detect"
    stores_with_603 = set()
    
    for _, _, filename, _, _, storeid, subcategory_id in image_paths:
        try:
            subcat = int(subcategory_id)
        except:
            continue
        if subcat == 603:
            stores_with_603.add(storeid)
    
    logger.info(f"Stores containing 603: {stores_with_603}")
    filtered_images = []
    
    for _, storename, filename, local_path, s3_key, storeid, subcategory_id in image_paths:
    
        try:
            subcat = int(subcategory_id)
        except:
            continue
    
        if storeid in stores_with_603:
            if subcat == 603:
                filtered_images.append(os.path.join(image_folder, filename))
        else:
            if subcat == 602:
                filtered_images.append(os.path.join(image_folder, filename))
    
    logger.info(f"Images selected after filtering: {len(filtered_images)}")
    
    if not filtered_images:
        logger.warning("No images match the store filtering logic — skipping YOLO inference.")
        return cyclecountid, False
    use_cache = False
    if os.path.exists(csv_output_path) and os.path.exists(predict_dir):
        try:
            df_check = pd.read_csv(csv_output_path)
            csv_images = set(df_check['imagefilename'].unique())
            filtered_set = set([os.path.basename(f) for f in filtered_images])
            if csv_images == filtered_set:
                use_cache = True
                for folder in sorted(os.listdir(predict_dir), reverse=True):
                    full_path = os.path.join(predict_dir, folder)
                    if os.path.isdir(full_path):
                        save_dir = full_path
                        break
                logger.info(f"Using cached predictions from: {csv_output_path} and {save_dir}")
        except Exception as e:
            logger.warning(f"Failed to check cached CSV: {e}. Regenerating predictions.")


    if not use_cache:
        logger.info("Regenerating YOLO predictions.")
        try:
            model = YOLO(model_path)
            config = load_config()
            cap_model_path = config['visicooler_config']['model_path']
            cap_model = YOLO(cap_model_path)
            cap_class_names = cap_model.names
            logger.info(f"Loaded CAP model from {cap_model_path}")
            logger.info(f"CAP model classes: {cap_class_names}")
            logger.info(f"Loaded YOLO model from {model_path}")
                        
            # -------------------------------
            # BASE MODEL → FRONTS
            # -------------------------------
            results = model.predict(source=filtered_images, conf=0.25, save=True)
            save_dir = results[0].save_dir if results else None
            logger.info(f"YOLO predictions saved to directory: {save_dir}")
            
            prediction_data = []
            
            for r in results:
                image_name = os.path.basename(r.path)
            
                fronts = []
                inferred_keys = set()  # prevent duplicate inferred SKUs
            
                # ---- parse BASE model detections (FRONTS ONLY) ----
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    class_name = class_names[class_id]
            
                    if should_ignore_class(class_id, class_names):
                        continue
            
                    if "cap" in class_name.lower():
                        continue
            
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    center_y = (y1 + y2) / 2
            
                    entry = {
                        'imagefilename': image_name,
                        'classid': class_id,
                        'classname': class_name,
                        'inference': conf,
                        'x1': x1,
                        'x2': x2,
                        'y1': y1,
                        'y2': y2,
                        'center_y': center_y
                    }
            
                    fronts.append(entry)
            
                    # ✅ always keep real fronts
                    prediction_data.append({
                        'imagefilename': image_name,
                        'classid': class_id,
                        'inference': conf,
                        'x1': x1,
                        'x2': x2,
                        'y1': y1,
                        'y2': y2
                    })
            
                # -------------------------------
                # CAP MODEL → CAPS ONLY
                # -------------------------------
                cap_results = cap_model.predict(source=[r.path], conf=0.1, save=False)
            
                for cr in cap_results:
                    for box in cr.boxes:
                        cap_class_id = int(box.cls[0])
                        cap_name = cap_class_names[cap_class_id]
            
                        if not is_cap_class(cap_name):
                            continue
            
                        cap_brand = extract_brand(cap_name)
                        if not cap_brand:
                            continue
            
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        center_y = (y1 + y2) / 2
            
                        # ---- nearest FRONT of same brand ----
                        nearest = None
                        bestd = 1e9
                        for f in fronts:
                            if extract_brand(f['classname']) != cap_brand:
                                continue
                            d = abs(f['center_y'] - center_y)
                            if d < bestd:
                                bestd = d
                                nearest = f
            
                        if not nearest:
                            continue
            
                        inferred_size = extract_size(nearest['classname'])
                        if not inferred_size:
                            continue
            
                        inferred_classid = None
                        for cid, cname in class_names.items():
                            if cap_brand in cname.lower() and inferred_size in cname:
                                inferred_classid = cid
                                break
            
                        if inferred_classid is None:
                            continue
            
                        dedup_key = (image_name, inferred_classid)
                        if dedup_key in inferred_keys:
                            continue
                        inferred_keys.add(dedup_key)
            
                        logger.info(
                            f"INFERRED (CAP) → {class_names[inferred_classid]} "
                            f"from cap {cap_name}"
                        )
            
                        prediction_data.append({
                            'imagefilename': image_name,
                            'classid': inferred_classid,
                            'inference': min(conf, 0.15),  #capped confidence
                            'x1': x1,
                            'x2': x2,
                            'y1': y1,
                            'y2': y2
                        })

            df = pd.DataFrame(prediction_data)
            df.to_csv(csv_output_path, index=False)
            logger.info(f"Predictions saved to CSV: {csv_output_path}")
        except Exception as e:
            logger.error(f"Failed to run YOLO predictions: {e}")
            raise

    if save_dir:
        for filename in os.listdir(save_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                local_path = os.path.join(save_dir, filename)
                s3_key = f"{s3_cycle_folder}/{filename}"
                try:
                    s3_handler.upload_file_to_s3(local_path, s3_key)
                    logger.info(f"Uploaded annotated image to S3: {s3_key}")
                except Exception as e:
                    logger.error(f"Failed to upload {filename} to S3: {e}")

    database_success = True
    try:
        clear_cyclecount_staging(cur, cyclecountid)
        conn.commit()
    except Exception as e:
        logger.error(f"Skipping database insert due to failure in clearing records: {e}")
        database_success = False

    if database_success:
        try:
            # Get max rowid for this cyclecountid
            cur.execute("SELECT MAX(rowid) FROM orgi.cyclecount_staging WHERE cyclecountid = %s", (cyclecountid,))
            result = cur.fetchone()
            rowid_counter = (result[0] if result[0] is not None else 0) + 1

            df = pd.read_csv(csv_output_path)
            now = datetime.now()
            for _, row in df.iterrows():
                imagefilename = row['imagefilename']
                s3path_actual_file = None
                storename = None
                storeid = None
                for filesequenceid, s_name, fname, _, s3_key, s_id, subcategory_id in image_paths:
                    if fname == imagefilename:
                        s3path_actual_file = s3_key
                        storename = s_name
                        storeid = s_id
                        break
                if not s3path_actual_file:
                    logger.warning(f"No s3path_actual_file found for {imagefilename}, skipping.")
                    continue
                s3path_annotated_file = f"{s3_cycle_folder}/{imagefilename}"
                insert_yolo_prediction(
                    cur=cur,
                    modelname=modelname,
                    imagefilename=imagefilename,
                    classid=int(row['classid']),
                    inference=float(row['inference']),
                    modelrun=now,
                    cyclecountid=cyclecountid,
                    rowid=rowid_counter,
                    s3_bucket_name=s3_bucket_name,
                    s3path_actual_file=s3path_actual_file,
                    s3path_annotated_file=s3path_annotated_file,
                    x1=float(row['x1']),
                    x2=float(row['x2']),
                    y1=float(row['y1']),
                    y2=float(row['y2']),
                    storename=storename,
                    storeid=storeid
                )
                rowid_counter += 1
            conn.commit()
            logger.info(f"Inserted {rowid_counter - 1} YOLO predictions into orgi.cyclecount_staging.")
        except Exception as e:
            logger.error(f"Failed to insert predictions to database: {e}")
            database_success = False
            conn.rollback()

    logger.info(f"YOLO predictions completed for cyclecountid {cyclecountid}. Database operations {'successful' if database_success else 'skipped'}.")
    return cyclecountid, database_success
