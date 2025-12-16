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
            if val is None:
                return None
            s = str(val).strip()
            s = s.replace(',', '')
            if s.endswith('.0'):
                s = s[:-2]
            m = re.search(r'(\d+)', s)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
            return None
        store_images = {}  
        filename_to_row = {}
        for row in image_paths:
            fileseqid, storename, filename, local_path, s3_key, orig_storeid, subcategory_id = row
            canonical_storeid = _get_canonical_storeid(filename, orig_storeid)
            sid = _norm_storeid(canonical_storeid)
            if sid is None:
                sid = None
            subcat_norm = normalize_subcat(subcategory_id)
            stored_row = (fileseqid, storename, filename, local_path, s3_key, canonical_storeid, subcat_norm, subcategory_id)
            store_images.setdefault(sid, []).append(stored_row)
            filename_to_row[filename] = stored_row

        store_target = {}  
        for sid, rows in store_images.items():
            has_603 = any(r[6] == 603 for r in rows)
            if has_603:
                store_target[sid] = 603
                continue
            has_602 = any(r[6] == 602 for r in rows)
            if has_602:
                store_target[sid] = 602
                continue
            store_target[sid] = None

        logger.info(f"Store targets (603 else 602): { {k: v for k,v in store_target.items() if v is not None} }")

        cur.execute("SELECT COALESCE(MAX(iterationid), 0) FROM orgi.coolermetricsmaster")
        row = cur.fetchone()
        current_iteration = row[0] + 1  # new batch iteration id
        logger.info(f"Using iteration ID for this batch: {current_iteration}")

        for sid, rows in store_images.items():
            target = store_target.get(sid) 
            for stored_row in rows:
                fileseqid, storename, filename, local_path, s3_key, canonical_storeid, subcat_norm, original_subcat_raw = stored_row
                if target is not None:
                    if subcat_norm != target:
                        logger.info(f"Skipping {filename} — store {sid} target={target}, file subcat={subcat_norm}")
                        logger.warning(f"ACTUAL SKIP EXECUTED → {filename} (sid={sid}, subcat={subcat_norm})")
                        continue
                else:
                    if subcat_norm is None:
                        logger.warning(f"Processing {filename} for store {sid} (no 603/602 found for store). Raw subcat='{original_subcat_raw}'")
                try:
                    iterationid = current_iteration
                    # use fileseqid as a base unique tran id for this 
                    iterationtranid = int(fileseqid) if (fileseqid is not None and str(fileseqid).isdigit()) else None
                    if iterationtranid is None:
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
                            if shelf_model.names.get(cls_id, "").lower() in ("shelf", "shelfs", "shelves"):
                                _, y1, _, y2 = box.xyxy[0]
                                y1, y2 = map(int, box.xyxy[0][1::2])
                                shelves.append({"top_y": y1, "bottom_y": y2})

                    merged_shelves = merge_overlapping_boxes(shelves)
                    shelf_regions = []
                    
                    num_bars = len(merged_shelves)
                    
                    # Case 0: no shelf bars → single shelf region
                    if num_bars == 0:
                        shelf_regions.append({
                            "shelf_id": 1,
                            "top": 0,
                            "bottom": image_height
                        })
                    
                    else:
                        shelf_id = 1
                    
                        # Top region (products can sit on top of first shelf bar)
                        shelf_regions.append({
                            "shelf_id": shelf_id,
                            "top": 0,
                            "bottom": merged_shelves[0]["top_y"]
                        })
                        shelf_id += 1
                    
                        # Regions between consecutive shelf bars
                        for prev_bar, next_bar in zip(merged_shelves[:-1], merged_shelves[1:]):
                            shelf_regions.append({
                                "shelf_id": shelf_id,
                                "top": prev_bar["bottom_y"],
                                "bottom": next_bar["top_y"]
                            })
                            shelf_id += 1
                    
                        # Bottom region ONLY for small coolers (≤ 2 bars)
                        if num_bars <= 2:
                            shelf_regions.append({
                                "shelf_id": shelf_id,
                                "top": merged_shelves[-1]["bottom_y"],
                                "bottom": image_height
                            })


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
                    cap_keywords = ["cap", "glass cap", "glasscap", "bottle cap", "bottlecap"]
                    cap_class_ids = set()
                    for cid, cname in class_names.items():
                        lname = cname.lower()
                        if "cap" in lname or "glasscap" in lname or "glass cap" in lname:
                            cap_class_ids.add(int(cid))
                    if not cap_class_ids:
                        cap_class_ids = {
                            29, 30, 34, 35, 45, 47, 48, 55, 56,
                            78, 80, 81, 86, 101, 111, 112
                        }
                    
                    front_skus = []
                    cap_detections = []
                    for sku in sku_detections:
                        if sku["class_id"] in cap_class_ids:
                            cap_detections.append(sku)
                        else:
                            front_skus.append(sku)
                    
                    shelf_buckets = {}
                    for region in shelf_regions:
                        shelf_buckets[region["shelf_id"]] = []
                    for f in front_skus:
                        assigned = False
                        for region in shelf_regions:
                            if region["top"] <= f["center_y"] < region["bottom"]:
                                shelf_buckets[region["shelf_id"]].append(f)
                                assigned = True
                                break
                        if not assigned:
                            best = None; bestd = 1e9
                            for region in shelf_regions:
                                mid = (region["top"] + region["bottom"]) / 2.0
                                d = abs(f["center_y"] - mid)
                                if d < bestd:
                                    bestd = d; best = region["shelf_id"]
                            shelf_buckets[best].append(f)
                    
                    def determine_brand_from_cap_label(label):
                        lname = label.lower()
                        if "coke" in lname or "coca" in lname:
                            return "coke"
                        if "sprite" in lname:
                            return "sprite"
                        if "fanta" in lname:
                            return "fanta"
                        if "kinley" in lname:
                            return "kinley"
                        if "pepsi" in lname:
                            return "pepsi"
                        # fallback
                        return "other"
                    
                    brand_default_pet = {
                        "coke": 13,    # Coke 1000ml PET
                        "sprite": 102, # Sprite 1000ml PET
                        "fanta": 37,   # Fanta Orange 1000ml PET
                        "kinley": 53,  # Kinley Water 1000ml PET
                        "pepsi": 87,   # Pepsi 1000ml PET (if needed)
                        "other": 82    # Other PET
                    }
                    brand_glass_equivalent = {
                        "coke": 17,    # Coke 250ml Glass
                        "sprite": 106, # Sprite 250ml Glass
                        "fanta": 41,   # Fanta 250ml Glass
                        "kinley": 49   # Kinley Soda 250ml Glass
                    }
                    
                    def find_classid_for_brand_size(brand, size_token):
                        brand = brand.lower()
                        for cid, cname in class_names.items():
                            cl = cname.lower()
                            if brand in cl and size_token in cl:
                                return int(cid)
                        return None
                    
                    inferred_skus = []
                    
                    for cap in cap_detections:
                        cap_label = class_names.get(cap["class_id"], "").lower()
                        cap_brand = determine_brand_from_cap_label(cap_label)
                        cap_is_glass = ("glass" in cap_label)
                    
                        # Determine shelf
                        cap_shelf = None
                        for region in shelf_regions:
                            if region["top"] <= cap["center_y"] <= region["bottom"]:
                                cap_shelf = region["shelf_id"]
                                break
                    
                        if cap_shelf is None:
                            # fallback to nearest shelf
                            best = None; bestd = 1e9
                            for region in shelf_regions:
                                mid = (region["top"] + region["bottom"]) / 2.0
                                d = abs(cap["center_y"] - mid)
                                if d < bestd:
                                    bestd = d; best = region["shelf_id"]
                            cap_shelf = best
                    
                        # Find nearest front SKU ONLY to infer SIZE (not for suppression)
                        candidate_fronts = shelf_buckets.get(cap_shelf, [])
                        closest_front = None
                        bestd = 1e9
                        for front in candidate_fronts:
                            d = abs(front["center_y"] - cap["center_y"])
                            if d < bestd:
                                bestd = d; closest_front = front
                    
                        # Infer class
                        inferred_class = None
                    
                        if cap_is_glass:
                            inferred_class = brand_glass_equivalent.get(
                                cap_brand,
                                brand_default_pet.get(cap_brand, brand_default_pet["other"])
                            )
                        else:
                            size_token = None
                            if closest_front:
                                front_name = closest_front["name"].lower()
                                for tok in ["2250", "1500", "1000", "750", "700", "600", "500", "250", "175"]:
                                    if tok in front_name:
                                        size_token = tok
                                        break
                            if size_token:
                                inferred_class = find_classid_for_brand_size(cap_brand, size_token)
                    
                            if inferred_class is None:
                                inferred_class = brand_default_pet.get(cap_brand, brand_default_pet["other"])
                    
                        if inferred_class is None:
                            continue
                    
                        # ✅ ALWAYS ADD
                        inferred_skus.append({
                            "class_id": int(inferred_class),
                            "name": class_names[int(inferred_class)],
                            "conf": cap["conf"],
                            "bbox": cap["bbox"],
                            "center_y": cap["center_y"],
                            "inferred": True
                        })
                    def extract_brand(name: str):
                        lname = name.lower()
                        if "coke" in lname or "coca" in lname:
                            return "coke"
                        if "sprite" in lname:
                            return "sprite"
                        if "fanta" in lname:
                            return "fanta"
                        if "kinley" in lname:
                            return "kinley"
                        if "pepsi" in lname:
                            return "pepsi"
                        return "other"
                    # =========================================
                    # PER-SHELF, PER-CAP BRAND-STRICT LOGIC
                    # =========================================
                    
                    final_skus = []
                    
                    for region in shelf_regions:
                        shelf_id = region["shelf_id"]
                    
                        shelf_fronts = [
                            f for f in front_skus
                            if region["top"] <= f["center_y"] <= region["bottom"]
                        ]
                    
                        shelf_caps = [
                            c for c in cap_detections
                            if region["top"] <= c["center_y"] <= region["bottom"]
                        ]
                    
                        # ---- group fronts & caps by brand ----
                        fronts_by_brand = defaultdict(list)
                        for f in shelf_fronts:
                            fronts_by_brand[extract_brand(f["name"])].append(f)
                    
                        caps_by_brand = defaultdict(list)
                        for c in shelf_caps:
                            caps_by_brand[extract_brand(c["name"])].append(c)
                    
                        all_brands = set(fronts_by_brand.keys()) | set(caps_by_brand.keys())
                    
                        for brand in all_brands:
                            fronts = fronts_by_brand.get(brand, [])
                            caps = caps_by_brand.get(brand, [])
                    
                            final_count = max(len(fronts), len(caps))
                            if len(caps) > len(fronts):
                                logger.warning(
                                    f"SHELF {shelf_id} | BRAND {brand} | CAPS > FRONTS ({len(caps)} > {len(fronts)})"
                                )

                            logger.info(
                                f"SHELF {shelf_id} | BRAND {brand} | "
                                f"fronts={len(fronts)} caps={len(caps)} → final={final_count}"
                            )
                    
                            chosen_sku = None
                    
                            # Prefer FRONT SKU (real detection)
                            if fronts:
                                chosen = sorted(fronts, key=lambda x: x["conf"], reverse=True)[0]
                                chosen_sku = {
                                    "class_id": chosen["class_id"],
                                    "name": chosen["name"],
                                    "conf": chosen["conf"],
                                    "bbox": chosen["bbox"],
                                    "center_y": chosen["center_y"],
                                    "inferred": False
                                }
                    
                            # Fallback to inferred SKU
                            else:
                                inferred = next(
                                    (i for i in inferred_skus if extract_brand(i["name"]) == brand),
                                    None
                                )
                                if inferred:
                                    chosen_sku = inferred
                    
                            if chosen_sku is None:
                                logger.warning(f"SHELF {shelf_id} | BRAND {brand} → NO SKU FOUND")
                                continue
                    
                            # 🔥 ADD EXACT COUNT
                            for i in range(final_count):
                                sku_copy = chosen_sku.copy()
                                sku_copy["center_y"] = (
                                    region["top"] + region["bottom"]
                                ) // 2
                                final_skus.append(sku_copy)

                    
                    # =========================================
                    # END FINAL PER-BRAND LOGIC
                    # =========================================

                    sku_detections = final_skus

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
                    for shelf_id, sku_list in shelf_sku_map.items():
                        counter = defaultdict(int)
                    
                        for sku in sku_list:
                            counter[sku["name"]] += 1
                    
                        logger.info(f"FINAL COUNT — Shelf {shelf_id}")
                        for name, count in counter.items():
                            logger.info(f"    {count} × {name}")
                    # plotting / annotated image (unchanged)
                    try:
                        if len(sku_results) > 0:
                            rendered_image = shelf_results[0].plot()
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

