import os
import cv2
from ultralytics import YOLO
import logging
from app.db_handler import initialize_db_connection, close_db_connection, get_classtext
from app.config_loader import load_config
from app.s3_handler import S3Handler
from datetime import datetime

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


def upload_to_visibilitydetails(conn, cur, records, cyclecountid):
    try:
        coolermetricstransaction_query = """
        INSERT INTO coolermetricstransaction
        (iterationid, iterationtranid, shelfnumber, productsequenceno, productclassid, x1, x2, y1, y2, confidence)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """

        coolermetricsmaster_query = """
        INSERT INTO coolermetricsmaster
        (iterationid, iterationtranid, storeid, caserid, modelrun, processedflag)
        VALUES (%s,%s,%s,%s,%s,%s)
        """

        transaction_records = []
        master_records = []

        for record in records:
            (filename, shelf_regions, shelf_sku_map) = record

            cur.execute(
                "SELECT storeid FROM batchtransactionvisibilityitems WHERE imagefilename = %s LIMIT 1",
                (filename,)
            )
            row = cur.fetchone()
            storeid = row[0] if row else None

            iterationid = records.index(record) + 1
            iterationtranid = 0
            caserid = len(shelf_regions)

            for region in shelf_regions:
                shelf_id = region["shelf_id"]
                productsequenceno = 0

                for sku in shelf_sku_map[shelf_id]:
                    productsequenceno += 1
                    iterationtranid += 1
                    x1, y1, x2, y2 = sku["bbox"]
                    productclassid = sku["class_id"]
                    confidence = sku["conf"]

                    transaction_records.append((
                        iterationid,
                        iterationtranid,
                        shelf_id,
                        productsequenceno,
                        productclassid,
                        x1,
                        x2,
                        y1,
                        y2,
                        confidence
                    ))

            master_records.append((
                iterationid,
                iterationtranid,
                storeid,
                caserid,
                datetime.now(),
                'N'
            ))

        if transaction_records:
            cur.executemany(coolermetricstransaction_query, transaction_records)

        if master_records:
            cur.executemany(coolermetricsmaster_query, master_records)

        conn.commit()
        logger.info("Inserted into coolermetricstransaction and coolermetricsmaster")

    except Exception as e:
        logger.error(f"DB insert failed: {e}")
        conn.rollback()
        raise


def run_visicooler_analysis(image_paths, config, s3_handler, conn, cur, output_folder_path, cyclecountid):
    try:
        shelf_model_path = config['visicooler_config']['caps_model_path']
        sku_model_path = config['yolo_config']['model_path']
        shelf_class_id = config['visicooler_config']['shelf_class_id']
        conf_threshold = config['visicooler_config']['conf_threshold']

        shelf_model = YOLO(shelf_model_path)
        sku_model = YOLO(sku_model_path)

        bulk_records = []

        for filesequenceid, storename, filename, local_path, s3_key, storeid in image_paths:
            try:
                image = cv2.imread(local_path)
                if image is None:
                    continue

                image_height, image_width = image.shape[:2]

                shelf_results = shelf_model(local_path, conf=conf_threshold)
                shelves = []
                for result in shelf_results:
                    scale_w = image_width / result.orig_shape[1]
                    scale_h = image_height / result.orig_shape[0]
                    for box in result.boxes:
                        if int(box.cls[0]) == shelf_class_id:
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1 * scale_w), int(y1 * scale_h), int(x2 * scale_w), int(y2 * scale_h)
                            shelves.append({"top_y": y1, "bottom_y": y2})

                merged_shelves = merge_overlapping_boxes(shelves)

                shelf_regions = []
                for i in range(len(merged_shelves)):
                    if i == 0:
                        shelf_regions.append({"shelf_id": i + 1, "top": 0, "bottom": merged_shelves[i]["bottom_y"]})
                    elif i == len(merged_shelves) - 1:
                        shelf_regions.append({"shelf_id": i + 1, "top": merged_shelves[i - 1]["top_y"], "bottom": image_height})
                    else:
                        shelf_regions.append({"shelf_id": i + 1, "top": merged_shelves[i - 1]["top_y"], "bottom": merged_shelves[i]["bottom_y"]})

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
                            "name": name,
                            "conf": float(box.conf[0]),
                            "bbox": (x1, y1, x2, y2),
                            "center_y": center_y,
                            "class_id": cls_id
                        })

                shelf_sku_map = {region["shelf_id"]: [] for region in shelf_regions}

                for sku in sku_detections:
                    for region in shelf_regions:
                        if region["top"] <= sku["center_y"] <= region["bottom"]:
                            shelf_sku_map[region["shelf_id"]].append(sku)
                            break

                bulk_records.append((filename, shelf_regions, shelf_sku_map))

                rendered_image = sku_results[0].plot()
                for region in shelf_regions:
                    cv2.line(rendered_image, (0, region["top"]), (image_width, region["top"]), (0, 255, 0), 2)
                    cv2.line(rendered_image, (0, region["bottom"]), (image_width, region["bottom"]), (0, 0, 255), 2)

                output_path = os.path.join(output_folder_path, f"segmented_{filename}")
                cv2.imwrite(output_path, rendered_image)

                s3_key_annotated = f"ModelResults/Visicooler_{cyclecountid}/segmented_{filename}"
                s3_handler.upload_file_to_s3(output_path, s3_key_annotated)

            except Exception as e:
                logger.error(f"Error processing image {filename}: {e}")
                continue

        if bulk_records:
            upload_to_visibilitydetails(conn, cur, bulk_records, cyclecountid)
        else:
            logger.warning("No visicooler records to upload.")

        return bulk_records

    except Exception as e:
        logger.error(f"Error in visicooler analysis: {e}")
        raise
