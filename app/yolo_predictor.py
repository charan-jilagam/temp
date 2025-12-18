import os
import yaml
import pandas as pd
import logging
from ultralytics import YOLO
from datetime import datetime
import ultralytics
import torch
from app.config_loader import load_config
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Utility helpers (UNCHANGED)
# --------------------------------------------------

def should_ignore_class(cls_id: int, class_names: list) -> bool:
    name = str(class_names[cls_id]).lower() if cls_id < len(class_names) else ""
    return any(k in name for k in ["700ml", "750ml"])

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
    if any(x in lname for x in ["bottle", "pet", "glass", "can", "cooler", "visicooler", "shelf"]):
        return False
    return "cap" in lname

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run_yolo_predictions(
    yaml_path,
    model_path,
    image_folder,
    csv_output_path,
    modelname,
    s3_bucket_name,
    s3_folder,
    conn,
    cur,
    s3_handler,
    image_paths,
    cyclecountid_override=None
):
    logger.info(f"Ultralytics version: {ultralytics.__version__}")
    logger.info(f"Torch version: {torch.__version__}")

    from app.db_handler import (
        get_max_cyclecountid,
        clear_cyclecount_staging,
        insert_yolo_prediction
    )

    # --------------------------------------------------
    # Cyclecount handling (UNCHANGED)
    # --------------------------------------------------
    if cyclecountid_override is not None:
        cyclecountid = cyclecountid_override
    else:
        cyclecountid = get_max_cyclecountid(cur) + 1

    s3_cycle_folder = f"ModelResults/CycleCount_{cyclecountid}"

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------
    with open(yaml_path, 'r') as f:
        yaml.safe_load(f)

    base_model = YOLO(model_path)
    class_names = base_model.names

    config = load_config()
    cap_model_path = config['visicooler_config']['model_path']
    cap_model = YOLO(cap_model_path)
    cap_class_names = cap_model.names

    # --------------------------------------------------
    # FILTER IMAGES → ONLY SUBCATEGORY 605
    # --------------------------------------------------
    filtered_images = []

    for _, _, filename, _, _, _, subcategory_id in image_paths:
        try:
            if int(subcategory_id) == 605:
                filtered_images.append(os.path.join(image_folder, filename))
        except Exception:
            continue

    logger.info(f"Images selected for inference (605 only): {len(filtered_images)}")

    if not filtered_images:
        logger.warning("No 605 images found — skipping inference.")
        return cyclecountid, False

    # --------------------------------------------------
    # Store-level counters (NEW)
    # --------------------------------------------------
    store_total_counts = defaultdict(int)
    store_inferred_counts = defaultdict(int)

    # --------------------------------------------------
    # BASE MODEL → FRONT FACING SKUs
    # --------------------------------------------------
    results = base_model.predict(source=filtered_images, conf=0.25, save=True)
    save_dir = results[0].save_dir if results else None

    prediction_data = []

    for r in results:
        image_name = os.path.basename(r.path)

        # resolve storeid once per image
        storeid = None
        for _, _, fname, _, _, sid, _ in image_paths:
            if fname == image_name:
                storeid = sid
                break

        fronts = []
        inferred_keys = set()

        # -------- FRONT SKUs --------
        for box in r.boxes:
            cid = int(box.cls[0])
            cname = class_names[cid]

            if should_ignore_class(cid, class_names):
                continue
            if "cap" in cname.lower():
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            center_y = (y1 + y2) / 2
            conf = float(box.conf[0])

            fronts.append({
                "classname": cname,
                "center_y": center_y
            })

            prediction_data.append({
                "imagefilename": image_name,
                "classid": cid,
                "inference": conf,
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2
            })

            if storeid is not None:
                store_total_counts[storeid] += 1

        # -------- CAP MODEL → INFERENCE --------
        cap_results = cap_model.predict(source=[r.path], conf=0.1, save=False)

        for cr in cap_results:
            for box in cr.boxes:
                cap_cid = int(box.cls[0])
                cap_name = cap_class_names[cap_cid]

                if not is_cap_class(cap_name):
                    continue

                cap_brand = extract_brand(cap_name)
                if not cap_brand:
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0])
                center_y = (y1 + y2) / 2
                conf = float(box.conf[0])

                nearest = None
                bestd = 1e9
                for f in fronts:
                    if extract_brand(f["classname"]) != cap_brand:
                        continue
                    d = abs(f["center_y"] - center_y)
                    if d < bestd:
                        bestd = d
                        nearest = f

                if not nearest:
                    continue

                size = extract_size(nearest["classname"])
                if not size:
                    continue

                inferred_classid = None
                for cid, cname in class_names.items():
                    if cap_brand in cname.lower() and size in cname:
                        inferred_classid = cid
                        break

                if inferred_classid is None:
                    continue

                key = (image_name, inferred_classid)
                if key in inferred_keys:
                    continue
                inferred_keys.add(key)

                prediction_data.append({
                    "imagefilename": image_name,
                    "classid": inferred_classid,
                    "inference": min(conf, 0.1),
                    "x1": x1,
                    "x2": x2,
                    "y1": y1,
                    "y2": y2
                })

                if storeid is not None:
                    store_total_counts[storeid] += 1
                    store_inferred_counts[storeid] += 1

    # --------------------------------------------------
    # STORE-LEVEL LOGGING (NEW)
    # --------------------------------------------------
    logger.info("===== STORE-LEVEL INFERENCE SUMMARY =====")
    for sid in sorted(store_total_counts.keys()):
        inferred = store_inferred_counts.get(sid, 0)
        total = store_total_counts.get(sid, 0)
        logger.info(
            f"STORE {sid} | Inferred (caps): {inferred} | "
            f"Total products: {total} | Ratio: {inferred}/{total}"
        )
    logger.info("========================================")

    # --------------------------------------------------
    # CSV OUTPUT (UNCHANGED)
    # --------------------------------------------------
    df = pd.DataFrame(prediction_data)
    df.to_csv(csv_output_path, index=False)

    # --------------------------------------------------
    # S3 UPLOAD (UNCHANGED)
    # --------------------------------------------------
    if save_dir:
        for fn in os.listdir(save_dir):
            if fn.lower().endswith(('.jpg', '.png', '.jpeg')):
                s3_handler.upload_file_to_s3(
                    os.path.join(save_dir, fn),
                    f"{s3_cycle_folder}/{fn}"
                )

    # --------------------------------------------------
    # DB INSERTS → orgi.cyclecount_staging (UNCHANGED)
    # --------------------------------------------------
    clear_cyclecount_staging(cur, cyclecountid)
    conn.commit()

    cur.execute(
        "SELECT MAX(rowid) FROM orgi.cyclecount_staging WHERE cyclecountid = %s",
        (cyclecountid,)
    )
    rowid = (cur.fetchone()[0] or 0) + 1

    now = datetime.now()

    for _, row in df.iterrows():
        imagefilename = row["imagefilename"]

        for _, storename, fname, _, s3_key, storeid, _ in image_paths:
            if fname == imagefilename:
                insert_yolo_prediction(
                    cur=cur,
                    modelname=modelname,
                    imagefilename=imagefilename,
                    classid=int(row["classid"]),
                    inference=float(row["inference"]),
                    modelrun=now,
                    cyclecountid=cyclecountid,
                    rowid=rowid,
                    s3_bucket_name=s3_bucket_name,
                    s3path_actual_file=s3_key,
                    s3path_annotated_file=f"{s3_cycle_folder}/{imagefilename}",
                    x1=float(row["x1"]),
                    x2=float(row["x2"]),
                    y1=float(row["y1"]),
                    y2=float(row["y2"]),
                    storename=storename,
                    storeid=storeid
                )
                rowid += 1
                break

    conn.commit()
    logger.info(f"YOLO predictions completed for cyclecountid {cyclecountid}")
    return cyclecountid, True
