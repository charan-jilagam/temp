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

        stores_with_603 = set()
        for _, _, filename, _, _, orig_storeid, subcategory_id in image_paths:
            canonical_storeid = _get_canonical_storeid(filename, orig_storeid)
            sid = _norm_storeid(canonical_storeid)
            try:
                subcat = int(subcategory_id)
            except Exception:
                try:
                    subcat = int(str(subcategory_id).strip())
                except Exception:
                    continue
            if sid is None:
                continue
            if subcat == 603:
                stores_with_603.add(sid)

        logger.info(f"Stores with 603 detected: {stores_with_603}")
        cur.execute("SELECT COALESCE(MAX(iterationid), 0) FROM orgi.coolermetricsmaster")
        row = cur.fetchone()
        current_iteration = row[0] + 1  # new batch iteration id
        logger.info(f"Using iteration ID for this batch: {current_iteration}")

        for filesequenceid, storename, filename, local_path, s3_key, orig_storeid, subcategory_id in image_paths:
            canonical_storeid = _get_canonical_storeid(filename, orig_storeid)
            sid = _norm_storeid(canonical_storeid)
            if sid is None:
                logger.warning(f"{filename}: invalid or missing storeid — skipping")
                continue
            try:
                subcat = int(subcategory_id)
            except Exception:
                try:
                    subcat = int(str(subcategory_id).strip())
                except Exception:
                    subcat = None
            should_skip = False
            if sid in stores_with_603:
                if subcat != 603:
                    should_skip = True
                    logger.info(f"Skipping {filename} — store {sid} has 603, ignoring {subcat}")
            else:
                if subcat != 602:
                    should_skip=True
                    logger.info(f"Skipping {filename} — store {sid} has no 603, only processing 602")
            if should_skip:
                logger.warning(f"ACTUAL SKIP EXECUTED → {filename} (sid={sid}, subcat={subcat})")
                continue
            try:
                iterationid = current_iteration

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

                shelf_results = shelf_model(local_path, conf=conf_threshold)
                shelves = []

                for result in shelf_results:
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
                num_shelves = len(merged_shelves)

                if num_shelves == 0:
                    logger.warning(f"No shelves detected for {filename}, using full image as single shelf")
                    shelf_regions = [{"shelf_id": 1, "top": 0, "bottom": image_height}]
                    num_shelves = 1
                else:
                    shelf_regions = []
                    shelf_id = 1
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
                    num_shelves = len(shelf_regions)

                logger.info(f"SHELVES FOUND: {num_shelves}")
                logger.info(f"SHELF REGIONS: {shelf_regions}")

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
                                    caserid = row2[0]      # mapped caserid (1,2,3,...)
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

                sku_results = sku_model(local_path, conf=0.35)

                sku_detections = []

                for result in sku_results:
                    scale_w = image_width / result.orig_shape[1]
                    scale_h = image_height / result.orig_shape[0]
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
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

                iterationtranid = 1

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
                            iterationid,
                            iterationtranid,
                            final_storeid,
                            caserid,
                            datetime.now()
                        ))
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
                        iterationtranid += 1
                        productsequenceno += 1

                conn.commit()
                logger.info(f"✅ Inserted for iteration {iterationid} : {iterationtranid - 1} products")

            except Exception as e:
                logger.error(f"Error processing image {filename}: {e}")
                conn.rollback()
                continue

        return bulk_records

    except Exception as e:
        logger.error(f"Error in visicooler analysis: {e}")
        raise
