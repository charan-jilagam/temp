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

    shelf_model = YOLO(config['visicooler_config']['caps_model_path'])
    sku_model   = YOLO(config['yolo_config']['model_path'])

    sku_class_names   = sku_model.names
    shelf_class_names = shelf_model.names

    def extract_brand(name):
        lname = name.lower()
        for b in ["coke", "sprite", "fanta", "kinley", "pepsi"]:
            if b in lname:
                return b
        return None

    def extract_size_ml(name):
        m = re.search(r'(\d+)\s*ml', name.lower())
        return int(m.group(1)) if m else None

    def pick_default_sku_for_brand(brand):
        candidates = []
        for cid, cname in sku_class_names.items():
            if brand in cname.lower():
                size = extract_size_ml(cname) or 9999
                candidates.append((size, cid, cname))
        if not candidates:
            return None
        candidates.sort()
        size, cid, cname = candidates[0]
        return {
            "class_id": cid,
            "name": cname,
            "bbox": (0, 0, 0, 0),
            "conf": 0.1
        }

    # ---------------- GROUP BY STORE (605 ONLY) ----------------
    store_images = defaultdict(list)
    for row in image_paths:
        fileseqid, storename, filename, local_path, s3_key, storeid, subcat = row
        if str(subcat).strip() == "605":
            store_images[storeid].append(row)

    cur.execute("SELECT COALESCE(MAX(iterationid),0) FROM orgi.coolermetricsmaster")
    iterationid = cur.fetchone()[0] + 1

    for storeid, images in store_images.items():
        iterationtranid = int(datetime.now().timestamp() * 1000) % 100000000

        for shelf_index, row in enumerate(images, start=1):
            try:
                _, _, filename, local_path, _, _, _ = row
                image = cv2.imread(local_path)
                if image is None:
                    continue

                # ---------------- FRONT SKUs ----------------
                front_skus = []
                sku_results = sku_model(local_path, conf=0.3)

                for r in sku_results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        name = sku_class_names.get(cls_id, "")
                        if should_ignore_class(cls_id, sku_class_names):
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        front_skus.append({
                            "class_id": cls_id,
                            "name": name,
                            "brand": extract_brand(name),
                            "size": extract_size_ml(name),
                            "conf": float(box.conf[0]),
                            "bbox": (x1, y1, x2, y2)
                        })

                # ---------------- CAPS ----------------
                cap_counts = defaultdict(int)
                cap_results = shelf_model(local_path, conf=config['visicooler_config']['conf_threshold'])

                for r in cap_results:
                    for box in r.boxes:
                        name = shelf_class_names.get(int(box.cls[0]), "").lower()
                        brand = extract_brand(name)
                        if brand:
                            cap_counts[brand] += 1

                # ---------------- RECONCILIATION ----------------
                final_skus = []
                brands = set([s["brand"] for s in front_skus if s["brand"]]) | set(cap_counts)

                for brand in brands:
                    fronts = [s for s in front_skus if s["brand"] == brand]
                    front_count = len(fronts)
                    cap_count = cap_counts.get(brand, 0)

                    if cap_count > front_count:
                        final_count = cap_count
                        if fronts:
                            exemplar = min(
                                fronts,
                                key=lambda f: abs((f["size"] or 0) - (fronts[0]["size"] or 0))
                            )
                        else:
                            exemplar = pick_default_sku_for_brand(brand)
                        conf = 0.1
                    else:
                        final_count = front_count
                        exemplar = fronts[0] if fronts else None
                        conf = exemplar["conf"] if exemplar else 0.1

                    if exemplar is None:
                        continue

                    for _ in range(final_count):
                        final_skus.append({
                            "class_id": exemplar["class_id"],
                            "conf": conf,
                            "bbox": exemplar["bbox"]
                        })

                # ---------------- SAVE IMAGE ----------------
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

                # ---------------- DB INSERT ----------------
                productsequenceno = 1
                for sku in final_skus:
                    x1, y1, x2, y2 = sku["bbox"]

                    cur.execute("""
                        INSERT INTO orgi.coolermetricsmaster
                        (iterationid, iterationtranid, storeid, modelrun, processed_flag)
                        VALUES (%s, %s, %s, %s, 'N')
                        ON CONFLICT DO NOTHING
                    """, (iterationid, iterationtranid, storeid, datetime.now()))

                    cur.execute("""
                        INSERT INTO orgi.coolermetricstransaction
                        (iterationid, iterationtranid, shelfnumber,
                         productsequenceno, productclassid,
                         x1, x2, y1, y2, confidence)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, (
                        iterationid, iterationtranid, shelf_index,
                        productsequenceno, sku["class_id"],
                        x1, x2, y1, y2, sku["conf"]
                    ))

                    productsequenceno += 1

                conn.commit()
                logger.info(
                    f"Inserted store={storeid}, shelf={shelf_index}, products={len(final_skus)}"
                )

            except Exception as e:
                logger.error(f"Error processing image {filename}: {e}")
                conn.rollback()

    return []
