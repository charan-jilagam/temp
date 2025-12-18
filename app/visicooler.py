import os
import cv2
from ultralytics import YOLO
import logging
from app.db_handler import initialize_db_connection, close_db_connection, get_classtext
from app.config_loader import load_config
from app.s3_handler import S3Handler
from datetime import datetime
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def should_ignore_class(cls_id: int, class_names: dict) -> bool:
    name = class_names.get(cls_id, "").lower()
    ignore_keywords = ["700ml", "750ml", "visicooler", "cooler"]
    return any(keyword in name for keyword in ignore_keywords)

def merge_overlapping_boxes(boxes, threshold=3):
    # retained for engine compatibility (not used)
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b["top_y"])
    merged = [boxes[0]]
    for box in boxes[1:]:
        prev = merged[-1]
        if box["top_y"] <= prev["bottom_y"] + threshold:
            prev["bottom_y"] = max(prev["bottom_y"], box["bottom_y"])
        else:
            merged.append(box)
    return merged

def check_visibilitydetails_schema(cur):
    return True

def upload_to_visibilitydetails(conn, cur, records, cyclecountid):
    pass

def run_visicooler_analysis(image_paths, config, s3_handler, conn, cur, output_folder_path, cyclecountid):
    try:
        # --------------------------------------------------
        # Model declarations (BOTH MODELS ARE USED)
        # --------------------------------------------------
        shelf_model_path = config['visicooler_config']['caps_model_path']   # caps only
        sku_model_path   = config['yolo_config']['model_path']             # front SKUs only
        conf_threshold   = config['visicooler_config']['conf_threshold']

        shelf_model = YOLO(shelf_model_path)
        sku_model   = YOLO(sku_model_path)

        sku_class_names   = sku_model.names
        shelf_class_names = shelf_model.names

        def _norm_storeid(sid):
            if sid is None:
                return None
            if isinstance(sid, str):
                s = sid.strip()
                if s.isdigit():
                    return int(s)
                return s
            return sid

        def _get_canonical_storeid(filename, orig_storeid):
            canonical = orig_storeid
            try:
                cur.execute("""
                    SELECT storeid
                    FROM orgi.batchtransactionvisibilityitems
                    WHERE imagefilename = %s
                    LIMIT 1
                """, (filename,))
                row = cur.fetchone()
                if row and row[0] is not None:
                    canonical = row[0]
            except Exception:
                pass
            return canonical

        def normalize_subcat(val):
            if val is None:
                return None
            s = str(val).strip().replace(',', '')
            if s.endswith('.0'):
                s = s[:-2]
            m = re.search(r'(\d+)', s)
            return int(m.group(1)) if m else None

        # --------------------------------------------------
        # Group images by store
        # --------------------------------------------------
        store_images = {}
        for row in image_paths:
            fileseqid, storename, filename, local_path, s3_key, orig_storeid, subcategory_id = row
            canonical_storeid = _get_canonical_storeid(filename, orig_storeid)
            sid = _norm_storeid(canonical_storeid)
            subcat_norm = normalize_subcat(subcategory_id)
            store_images.setdefault(sid, []).append(
                (fileseqid, storename, filename, local_path, s3_key, canonical_storeid, subcat_norm)
            )

        logger.info("Store targets locked to subcategory 605")

        cur.execute("SELECT COALESCE(MAX(iterationid), 0) FROM orgi.coolermetricsmaster")
        iterationid = cur.fetchone()[0] + 1

        for sid, rows in store_images.items():
           # 🔑 generate ONCE per store
            iterationtranid = (
                int(rows[0][0])
                if rows[0][0] and str(rows[0][0]).isdigit()
                else int(datetime.now().timestamp() * 1000) % 100000000
            )
            for stored_row in rows:
                fileseqid, storename, filename, local_path, s3_key, final_storeid, subcat_norm = stored_row

                if subcat_norm != 605:
                    continue

                try:
                    image = cv2.imread(local_path)
                    if image is None:
                        continue

                    image_height, image_width = image.shape[:2]
                    os.makedirs(output_folder_path, exist_ok=True)

                    # --------------------------------------------------
                    # Single shelf region (605 = one image = one shelf)
                    # --------------------------------------------------
                    shelf_605_rows = [r for r in store_images[sid] if r[6] == 605]
                    shelf_index = shelf_605_rows.index(stored_row) + 1

                    shelf_regions = [{
                        "shelf_id": shelf_index,
                        "top": 0,
                        "bottom": image_height
                    }]

                    # --------------------------------------------------
                    # Caser mapping (unchanged)
                    # --------------------------------------------------
                    caserid = 0
                    try:
                        cur.execute("""
                            SELECT cooler FROM orgi.storemaster
                            WHERE storeid = %s LIMIT 1
                        """, (final_storeid,))
                        row = cur.fetchone()
                        if row and row[0]:
                            m = re.search(r'(\d+)', row[0])
                            if m:
                                cur.execute("""
                                    SELECT caserid FROM orgi.puritymapping
                                    WHERE casername ILIKE %s LIMIT 1
                                """, (f"%{m.group(1)}%",))
                                prow = cur.fetchone()
                                if prow:
                                    caserid = prow[0]
                    except Exception:
                        pass

                    # --------------------------------------------------
                    # FRONT SKU DETECTION (SKU MODEL ONLY)
                    # --------------------------------------------------
                    sku_results = sku_model(local_path, conf=0.3)
                    front_skus = []

                    for result in sku_results:
                        if not result.orig_shape:
                            continue
                        sw = image_width / result.orig_shape[1]
                        sh = image_height / result.orig_shape[0]

                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            if should_ignore_class(cls_id, sku_class_names):
                                continue

                            x1, y1, x2, y2 = box.xyxy[0]
                            front_skus.append({
                                "class_id": cls_id,
                                "name": sku_class_names[cls_id],
                                "conf": float(box.conf[0]),
                                "bbox": (
                                    int(x1 * sw), int(y1 * sh),
                                    int(x2 * sw), int(y2 * sh)
                                ),
                                "center_y": int((y1 + y2) * sh / 2)
                            })

                    # --------------------------------------------------
                    # CAP DETECTION (SHELF MODEL ONLY)
                    # --------------------------------------------------
                    cap_results = shelf_model(local_path, conf=conf_threshold)
                    cap_detections = []

                    for result in cap_results:
                        if not result.orig_shape:
                            continue
                        sw = image_width / result.orig_shape[1]
                        sh = image_height / result.orig_shape[0]

                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            name = shelf_class_names.get(cls_id, "").lower()
                            if "cap" not in name:
                                continue

                            x1, y1, x2, y2 = box.xyxy[0]
                            cap_detections.append({
                                "conf": float(box.conf[0]),
                                "center_y": int((y1 + y2) * sh / 2)
                            })

                    # --------------------------------------------------
                    # Cap → nearest front SKU inference
                    # --------------------------------------------------
                    def extract_brand(name):
                        lname = name.lower()
                        for b in ["coke", "sprite", "fanta", "kinley", "pepsi"]:
                            if b in lname:
                                return b
                        return "other"

                    inferred_skus = []

                    for cap in cap_detections:
                        closest = None
                        bestd = 1e9
                        for front in front_skus:
                            d = abs(front["center_y"] - cap["center_y"])
                            if d < bestd:
                                bestd = d
                                closest = front
                        if closest:
                            inferred_skus.append({
                                **closest,
                                "conf": cap["conf"],
                                "inferred": True
                            })

                    final_skus = front_skus + inferred_skus
                    shelf_sku_map = {shelf_index: final_skus}

                    # --------------------------------------------------
                    # Annotated image (SKU view)
                    # --------------------------------------------------
                    try:
                        rendered = sku_results[0].plot()
                        out = os.path.join(output_folder_path, f"segmented_{filename}")
                        cv2.imwrite(out, rendered)
                        s3_handler.upload_file_to_s3(
                            out,
                            f"ModelResults/Visicooler_{cyclecountid}/segmented_{filename}"
                        )
                    except Exception:
                        pass

                    # --------------------------------------------------
                    # DB INSERTS (UNCHANGED)
                    # --------------------------------------------------
                    for shelf_id, sku_list in shelf_sku_map.items():
                        productsequenceno = 1
                        for sku in sku_list:
                            x1, y1, x2, y2 = sku["bbox"]

                            cur.execute("""
                                INSERT INTO orgi.coolermetricsmaster
                                (iterationid, iterationtranid, storeid, caserid, modelrun, processed_flag)
                                VALUES (%s, %s, %s, %s, %s, 'N')
                                ON CONFLICT DO NOTHING
                            """, (
                                iterationid, iterationtranid,
                                final_storeid, caserid,
                                datetime.now()
                            ))

                            cur.execute("""
                                INSERT INTO orgi.coolermetricstransaction
                                (iterationid, iterationtranid, shelfnumber,
                                 productsequenceno, productclassid,
                                 x1, x2, y1, y2, confidence)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                iterationid, iterationtranid,
                                shelf_id, productsequenceno,
                                sku["class_id"],
                                x1, x2, y1, y2,
                                sku["conf"]
                            ))
                            productsequenceno += 1

                    conn.commit()
                    logger.info(
                        f"Inserted store={final_storeid}, shelf={shelf_index}, products={len(final_skus)}"
                    )

                except Exception as e:
                    logger.error(f"Error processing image {filename}: {e}")
                    conn.rollback()

        return []

    except Exception as e:
        logger.error(f"Fatal visicooler error: {e}")
        raise
