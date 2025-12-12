import os
import cv2
from ultralytics import YOLO
import logging
from app.db_handler import initialize_db_connection, close_db_connection, get_classtext
from app.config_loader import load_config
from app.s3_handler import S3Handler
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def should_ignore_class(cls_id: int, class_names: dict) -> bool:
    """
    Returns True if the predicted class should be ignored based on rules.
    Currently ignores any class whose name contains '700ml' or '750ml'.
    """
    name = class_names.get(cls_id, "").lower()
    ignore_keywords = ["700ml", "750ml"]

    return any(keyword in name for keyword in ignore_keywords)

def merge_overlapping_boxes(boxes, threshold=10):
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
        shelf_model_path = config['visicooler_config']['caps_model_path']
        sku_model_path = config['yolo_config']['model_path']
        shelf_class_id = config['visicooler_config']['shelf_class_id']
        conf_threshold = config['visicooler_config']['conf_threshold']

        shelf_model = YOLO(shelf_model_path)
        sku_model = YOLO(sku_model_path)
        class_names = sku_model.names
        bulk_records = []

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
            """Robust normalization for subcategory values -> returns int or None."""
            if val is None:
                return None
            s = str(val).strip()
            # remove commas, trailing .0 and stray chars
            s = s.replace(',', '')
            if s.endswith('.0'):
                s = s[:-2]
            # sometimes there's parentheses etc. keep only leading digits
            m = re.search(r'(\d+)', s)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
            return None

        # -------------------------
        # PHASE 0: Build store -> images mapping (pre-scan)
        # -------------------------
        store_images = {}  # sid -> list of image rows (keep original tuple)
        filename_to_row = {}  # helper map in case we need it
        for row in image_paths:
            fileseqid, storename, filename, local_path, s3_key, orig_storeid, subcategory_id = row
            canonical_storeid = _get_canonical_storeid(filename, orig_storeid)
            sid = _norm_storeid(canonical_storeid)
            if sid is None:
                # keep record but grouped under None so we can log later
                sid = None
            subcat_norm = normalize_subcat(subcategory_id)
            # store normalized values in a tuple variant so we don't re-parse later
            stored_row = (fileseqid, storename, filename, local_path, s3_key, canonical_storeid, subcat_norm, subcategory_id)
            store_images.setdefault(sid, []).append(stored_row)
            filename_to_row[filename] = stored_row

        # -------------------------
        # PHASE 1: Decide per-store target subcategory (603 if any, else 602 if any, else None)
        # -------------------------
        store_target = {}  # sid -> target_subcat (603/602/None)
        for sid, rows in store_images.items():
            has_603 = any(r[6] == 603 for r in rows)
            if has_603:
                store_target[sid] = 603
                continue
            has_602 = any(r[6] == 602 for r in rows)
            if has_602:
                store_target[sid] = 602
                continue
            # fallback: None -> you can choose to process any available image for this store
            store_target[sid] = None

        logger.info(f"Store targets (603 else 602): { {k: v for k,v in store_target.items() if v is not None} }")

        # get iteration id for this run
        cur.execute("SELECT COALESCE(MAX(iterationid), 0) FROM orgi.coolermetricsmaster")
        row = cur.fetchone()
        current_iteration = row[0] + 1  # new batch iteration id
        logger.info(f"Using iteration ID for this batch: {current_iteration}")

        # -------------------------
        # PHASE 2: Process images according to the decided target_subcat per store
        # -------------------------
        # We'll use the filesequenceid as the base for iterationtranid to avoid collisions
        for sid, rows in store_images.items():
            target = store_target.get(sid)  # 603, 602 or None (fallback)
            # if sid is None (no store id), just try to process rows individually
            for stored_row in rows:
                fileseqid, storename, filename, local_path, s3_key, canonical_storeid, subcat_norm, original_subcat_raw = stored_row
                # If a target was decided, only process rows that match it.
                # If target is None, we will process ANY row (fallback). If you want to skip fallback, change this behavior.
                if target is not None:
                    if subcat_norm != target:
                        logger.info(f"Skipping {filename} — store {sid} target={target}, file subcat={subcat_norm}")
                        logger.warning(f"ACTUAL SKIP EXECUTED → {filename} (sid={sid}, subcat={subcat_norm})")
                        continue
                else:
                    # fallback: if subcat is None we still try to process the image, but log warning
                    if subcat_norm is None:
                        logger.warning(f"Processing {filename} for store {sid} (no 603/602 found for store). Raw subcat='{original_subcat_raw}'")
                # now process the image
                try:
                    iterationid = current_iteration
                    # use fileseqid as a base unique tran id for this file.
                    # If fileseqid might collide across batches, consider mapping to an internal counter.
                    iterationtranid = int(fileseqid) if (fileseqid is not None and str(fileseqid).isdigit()) else None
                    if iterationtranid is None:
                        # fallback unique tran id: use a timestamp-based small int
                        iterationtranid = int(datetime.now().timestamp() * 1000) % 100000000

                    image = cv2.imread(local_path)
                    if image is None:
                        logger.error(f"Failed to load image: {local_path}")
                        continue

                    image_height, image_width = image.shape[:2]
                    os.makedirs(output_folder_path, exist_ok=True)

                    final_storeid = canonical_storeid
                    if final_storeid is None:
                        logger.error(f"StoreID not found for {filename}. Skipping DB inserts for this image.")
                        continue

                    # shelf detection
                    shelf_results = shelf_model(local_path, conf=conf_threshold)
                    shelves = []
                    for result in shelf_results:
                        # guard division by zero if orig_shape is malformed
                        if result.orig_shape is None or len(result.orig_shape) < 2 or result.orig_shape[0] == 0:
                            continue
                        scale_h = image_height / result.orig_shape[0]
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            try:
                                name = shelf_model.names[cls_id]
                            except Exception:
                                name = ""
                            if cls_id == shelf_class_id or "shelf" in name.lower():
                                _, y1, _, y2 = box.xyxy[0]
                                y1, y2 = int(y1 * scale_h), int(y2 * scale_h)
                                shelves.append({"top_y": y1, "bottom_y": y2})

                    merged_shelves = merge_overlapping_boxes(shelves)
                    if len(merged_shelves) == 0:
                        logger.warning(f"No shelves detected for {filename}, using full image as single shelf")
                        shelf_regions = [{"shelf_id": 1, "top": 0, "bottom": image_height}]
                    else:
                        shelf_regions = []
                        shelf_id = 1
                        # top region above first detected shelf
                        shelf_regions.append({"shelf_id": shelf_id, "top": 0, "bottom": merged_shelves[0]["top_y"]})
                        shelf_id += 1
                        for i in range(len(merged_shelves) - 1):
                            shelf_regions.append({
                                "shelf_id": shelf_id,
                                "top": merged_shelves[i]["bottom_y"],
                                "bottom": merged_shelves[i + 1]["top_y"]
                            })
                            shelf_id += 1
                        shelf_regions.append({"shelf_id": shelf_id, "top": merged_shelves[-1]["bottom_y"], "bottom": image_height})

                    logger.info(f"SHELVES FOUND: {len(shelf_regions)}")
                    logger.info(f"SHELF REGIONS: {shelf_regions}")

                    # caserid mapping (unchanged)
                    caserid = 0
                    try:
                        cur.execute("""
                            SELECT cooler
                            FROM orgi.storemaster
                            WHERE storeid = %s
                            LIMIT 1
                        """, (final_storeid,))
                        row = cur.fetchone()
                        if row and row[0]:
                            cooler_text = row[0]
                            match = re.search(r'(\d+)', cooler_text)
                            if match:
                                extracted_number = int(match.group(1))
                                try:
                                    cur.execute("""
                                        SELECT caserid 
                                        FROM orgi.puritymapping
                                        WHERE casername ILIKE %s
                                        LIMIT 1
                                    """, (f"%{extracted_number}%",))
                                    row2 = cur.fetchone()
                                    if row2:
                                        caserid = row2[0]
                                        logger.info(f"Mapped caser number {extracted_number} → puritymapping.caserid = {caserid}")
                                    else:
                                        caserid = 0
                                        logger.warning(f"No mapped caserid found in puritymapping for number {extracted_number}, using caserid = 0")
                                except Exception as e:
                                    logger.error(f"Failed mapping caserid using puritymapping for number {extracted_number}: {e}")
                                    caserid = 0
                            else:
                                logger.warning(f"No numeric value found in cooler text '{cooler_text}', using caserid = 0")
                                caserid = 0
                    except Exception as e:
                        logger.error(f"Failed to fetch or parse cooler size for store {final_storeid}: {e}")
                        caserid = 0

                    # SKU detection
                    sku_results = sku_model(local_path, conf=0.35)
                    sku_detections = []
                    for result in sku_results:
                        if result.orig_shape is None or len(result.orig_shape) < 2:
                            continue
                        scale_w = image_width / result.orig_shape[1]
                        scale_h = image_height / result.orig_shape[0]
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            if should_ignore_class(cls_id, class_names):
                                logger.info(f"Ignoring detection: {class_names.get(cls_id, str(cls_id))} (cls_id={cls_id})")
                                continue
                            name = sku_model.names[cls_id]
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1 * scale_w), int(y1 * scale_h), int(x2 * scale_w), int(y2 * scale_h)
                            center_y = (y1 + y2) // 2
                            sku_detections.append({
                                "class_id": cls_id,
                                "name": name,
                                "conf": float(box.conf[0]),
                                "bbox": (x1, y1, x2, y2),
                                "center_y": center_y
                            })

                    logger.info(f"TOTAL SKUs DETECTED: {len(sku_detections)}")
                    shelf_sku_map = {region["shelf_id"]: [] for region in shelf_regions}
                    for sku in sku_detections:
                        assigned = False
                        for region in shelf_regions:
                            if region["top"] <= sku["center_y"] <= region["bottom"]:
                                shelf_sku_map[region["shelf_id"]].append(sku)
                                assigned = True
                                break
                        if not assigned:
                            logger.warning(f"SKU NOT ASSIGNED TO ANY SHELF: {sku['name']} at y={sku['center_y']}")

                    # plotting / annotated image (unchanged)
                    try:
                        if len(sku_results) > 0:
                            rendered_image = sku_results[0].plot()
                            for region in shelf_regions:
                                cv2.line(rendered_image, (0, region["top"]), (image_width, region["top"]), (0, 255, 0), 2)
                                cv2.line(rendered_image, (0, region["bottom"]), (image_width, region["bottom"]), (0, 0, 255), 2)
                            output_path = os.path.join(output_folder_path, f"segmented_{filename}")
                            cv2.imwrite(output_path, rendered_image)
                            s3_key_annotated = f"ModelResults/Visicooler_{cyclecountid}/segmented_{filename}"
                            s3_handler.upload_file_to_s3(output_path, s3_key_annotated)
                            logger.info(f"Uploaded segmented image to S3: {s3_key_annotated}")
                    except Exception as e:
                        logger.error(f"Failed to generate/upload annotated image for {filename}: {e}")

                    # DB inserts: use iterationtranid derived from fileseqid so it is unique per image
                    for shelf_id, sku_list in shelf_sku_map.items():
                        productsequenceno = 1
                        for sku in sku_list:
                            x1, y1, x2, y2 = sku["bbox"]
                            # master row (safe ON CONFLICT)
                            cur.execute("""
                                INSERT INTO orgi.coolermetricsmaster
                                (iterationid, iterationtranid, storeid, caserid, modelrun, processed_flag)
                                VALUES (%s, %s, %s, %s, %s, 'N')
                                ON CONFLICT DO NOTHING
                            """, (
                                iterationid,
                                iterationtranid,
                                final_storeid,
                                caserid,
                                datetime.now()
                            ))
                            # transaction row
                            cur.execute("""
                                INSERT INTO orgi.coolermetricstransaction
                                (iterationid, iterationtranid, shelfnumber,
                                 productsequenceno, productclassid, x1, x2, y1, y2, confidence)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                iterationid,
                                iterationtranid,
                                shelf_id,
                                productsequenceno,
                                sku["class_id"],
                                x1,
                                x2,
                                y1,
                                y2,
                                sku["conf"]
                            ))
                            productsequenceno += 1

                    conn.commit()
                    logger.info(f"✅ Inserted for iteration {iterationid} , tranid {iterationtranid} : {sum(len(v) for v in shelf_sku_map.values())} products")

                except Exception as e:
                    logger.error(f"Error processing image {filename}: {e}")
                    conn.rollback()
                    continue

        return bulk_records
    except Exception as e:
        logger.error(f"Error in visicooler analysis: {e}")
        raise

