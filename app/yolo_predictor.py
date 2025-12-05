import os
import yaml
import pandas as pd
import logging
from ultralytics import YOLO
from datetime import datetime
import ultralytics
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        class_names = data['names']
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
            logger.info(f"Loaded YOLO model from {model_path}")
            results = model.predict(source=filtered_images, conf=0.35, save=True)
            save_dir = results[0].save_dir if results else None
            logger.info(f"YOLO predictions saved to directory: {save_dir}")

            prediction_data = []
            for r in results:
                image_name = os.path.basename(r.path)
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    prediction_data.append({
                        'imagefilename': image_name,
                        'classid': class_id,
                        'inference': conf,
                        'x1': float(box.xyxy[0][0]),
                        'x2': float(box.xyxy[0][2]),
                        'y1': float(box.xyxy[0][1]),
                        'y2': float(box.xyxy[0][3])
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
