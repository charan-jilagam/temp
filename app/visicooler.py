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

def check_visibilitydetails_schema(cur):
    """Check if all required columns exist in orgi.visibilitydetails with correct data types."""
    required_columns = {
        'cyclecountid': {'expected_types': ['integer', 'bigint']},
        'imagefilename': {'expected_types': ['varchar', 'text', 'character varying']},
        'numshelf': {'expected_types': ['integer', 'bigint']},
        'numproducts': {'expected_types': ['integer', 'bigint']},
        'numpureshelf': {'expected_types': ['integer', 'bigint']},
        'coolersize': {'expected_types': ['varchar', 'text', 'character varying']},
        'percentrgb': {'expected_types': ['double precision', 'numeric', 'real']},
        'chilleditems': {'expected_types': ['integer', 'bigint']},
        'warmitems': {'expected_types': ['integer', 'bigint']},
        'skus_detected': {'expected_types': ['varchar', 'text', 'character varying']},
        'share_chilled': {'expected_types': ['double precision', 'numeric', 'real']},
        'share_warm': {'expected_types': ['double precision', 'numeric', 'real']},
        'present_no_facings': {'expected_types': ['varchar', 'text', 'character varying']}
    }
    try:
        columns_list = ', '.join(f"'{col}'" for col in required_columns.keys())
        query = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'orgi' AND table_name = 'visibilitydetails'
            AND column_name IN ({columns_list})
        """
        cur.execute(query)
        columns = {row[0]: row[1].lower() for row in cur.fetchall()}

        for col, props in required_columns.items():
            if col not in columns:
                logger.error(
                    f"Column {col} not found in orgi.visibilitydetails. Please add it with: ALTER TABLE orgi.visibilitydetails ADD COLUMN {col} {props['expected_types'][0]};")
                return False
            if columns[col] not in props['expected_types']:
                logger.error(
                    f"Column {col} has unexpected data type: {columns[col]}. Expected one of {props['expected_types']}. Please alter with: ALTER TABLE orgi.visibilitydetails ALTER COLUMN {col} TYPE {props['expected_types'][0]};")
                return False
        return True
    except Exception as e:
        logger.error(f"Failed to check visibilitydetails schema: {e}")
        return False

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
    """Upload visicooler analysis results to coolermetrics tables."""
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

        for index, record in enumerate(records):
            (filename, num_shelves, total_products, pure_shelf_count, visicooler_size,
             percent_rgb, chilled_items, warm_items, skus_detected,
             share_chilled, share_warm, present_no_facings) = record

            iterationid = index + 1

            cur.execute(
                "SELECT storeid FROM batchtransactionvisibilityitems WHERE imagefilename = %s LIMIT 1",
                (filename,)
            )
            row = cur.fetchone()
            storeid = row[0] if row else None

            cur.execute(
                coolermetricsmaster_query,
                (
                    iterationid,
                    0,
                    storeid,
                    num_shelves,
                    datetime.now(),
                    'N'
                )
            )

        conn.commit()
        logger.info("Inserted into coolermetricsmaster")

    except Exception as e:
        logger.error(f"Failed to upload to coolermetrics tables: {e}")
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
