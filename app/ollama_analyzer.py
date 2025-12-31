# import os
# import json
# import csv
# import logging
# import re
# import requests
# from datetime import datetime
# from ollama import Client
# from ultralytics import YOLO

# from app.config_loader import load_config, load_json_classes
# from app.db_handler import (
#     initialize_db_connection,
#     close_db_connection,
#     get_classtext,
#     insert_ollama_results,
#     get_max_stagingid
# )

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)


# def check_ollama_server(ollama_host, model_name):
#     try:
#         r = requests.get(f"{ollama_host}/api/tags", timeout=5)
#         if r.status_code != 200:
#             return False
#         models = r.json().get("models", [])
#         return any(m["name"] == model_name for m in models)
#     except Exception as e:
#         logger.error(f"Ollama server check failed: {e}")
#         return False


# def extract_json(text):
#     if not text:
#         return {}
#     txt = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()
#     m = re.search(r"\{.*\}", txt, re.DOTALL)
#     if not m:
#         return {}
#     try:
#         return json.loads(m.group(0))
#     except Exception:
#         return {}


# def ollama_generate(ollama_host, model_name, prompt, image_path):
#     client = Client(host=ollama_host)
#     try:
#         r = client.chat(
#             model=model_name,
#             messages=[
#                 {"role": "system", "content": "Return ONLY a JSON object."},
#                 {"role": "user", "content": prompt, "images": [image_path]},
#             ],
#             format="json",
#             options={"temperature": 0}
#         )
#         return extract_json(r["message"]["content"])
#     except Exception as e:
#         logger.warning(f"Ollama retry: {e}")
#         try:
#             r = client.chat(
#                 model=model_name,
#                 messages=[
#                     {"role": "system", "content": "Return ONLY a JSON object."},
#                     {"role": "user", "content": prompt, "images": [image_path]},
#                 ],
#                 options={"temperature": 0}
#             )
#             return extract_json(r["message"]["content"])
#         except Exception:
#             return {}

# _ACTIVATION_YOLO = None

# def get_activation_yolo(model_path):
#     global _ACTIVATION_YOLO
#     if _ACTIVATION_YOLO is None:
#         logger.info(f"Loading activation YOLO: {model_path}")
#         _ACTIVATION_YOLO = YOLO(model_path)
#     return _ACTIVATION_YOLO


# def run_activation_yolo(image_path, model_path, conf_threshold=0.3):
#     """
#     Run activation YOLO model with configurable confidence threshold.
#     Returns set of detected class names (lowercased, underscored).
#     """
#     model = get_activation_yolo(model_path)
#     results = model(image_path, conf=conf_threshold, verbose=False)

#     detected = set()
#     for r in results:
#         for cls in r.boxes.cls:
#             class_name = model.names[int(cls)].lower().replace(" ", "_")
#             detected.add(class_name)
#             logger.info(f"Activation YOLO detected: {class_name} in {os.path.basename(image_path)}")
    
#     return detected

# def analyze_image(image_path, ollama_host, prompts, class_ids, model_name):
#     merged = {}

#     detect = ollama_generate(
#         ollama_host, model_name, prompts["visicooler_detect"], image_path
#     )
#     merged.update({str(k): v for k, v in detect.items()})

#     if merged.get("1001") == "N":
#         for k in range(1002, 1012):
#             merged[str(k)] = "N/A"
#     else:
#         attrs = ollama_generate(
#             ollama_host, model_name, prompts["visicooler_attrs"], image_path
#         )
#         merged.update({str(k): v for k, v in attrs.items()})

#         for p in ["extended_visibility_group1", "extended_visibility_group2"]:
#             merged.update(
#                 ollama_generate(ollama_host, model_name, prompts[p], image_path)
#             )

#     return {k: v for k, v in merged.items() if k in class_ids}

# def run_ollama_analysis(
#     image_paths,
#     image_folder,
#     output_csv,
#     config_path,
#     class_ids_path,
#     ollama_host,
#     s3_handler,
#     s3_annotated_folder,
#     db_config,
#     cyclecountid
# ):
#     config = load_config(config_path)
#     ollama_cfg = config["ollama_config"]
#     model_name = ollama_cfg["ollama_model"]
#     prompts = ollama_cfg["prompts"]
#     activation_yolo_model = ollama_cfg.get("activation_yolo_model")
    
#     # Configurable confidence threshold (default 0.3)
#     activation_conf_threshold = ollama_cfg.get("activation_conf_threshold", 0.3)

#     if not check_ollama_server(ollama_host, model_name):
#         logger.error("Ollama not available")
#         return [], None

#     class_ids = load_json_classes(class_ids_path)

#     conn, cur = initialize_db_connection(db_config)
#     stagingid = get_max_stagingid(cur) + 1

#     results = []
#     rowid = 1
    
#     # Stats tracking
#     total_images = 0
#     activation_processed = 0
#     ollama_processed = 0

#     # ========================================
#     # KEY OPTIMIZATION: Map class IDs once (O(1) lookups)
#     # ========================================
#     activation_mappings = {
#         "poster": "1019",
#         "dps": "1053",
#         "menu_board": "1023"
#     }

#     for (_, storename, filename, local_path, s3_key, storeid, subcategory_id) in image_paths:
#         if subcategory_id in [601, 602, 603, 604, 605]:
#             continue
#         if not os.path.exists(local_path):
#             continue

#         total_images += 1
#         logger.info(f"[{total_images}] Processing: {filename}")
        
#         # ========================================
#         # STEP 1: Run activation YOLO (FAST - ~50ms)
#         # ========================================
#         activation_detected = set()
#         if activation_yolo_model:
#             activation_detected = run_activation_yolo(
#                 local_path, 
#                 activation_yolo_model, 
#                 conf_threshold=activation_conf_threshold
#             )
#             if activation_detected:
#                 logger.info(f"   ✓ Activation YOLO detected: {activation_detected}")
#             else:
#                 logger.info(f"   ✗ Activation YOLO: No detections")

#         # ========================================
#         # STEP 2: Check if activation model can handle this image
#         # OLD CODE: Would run Ollama regardless
#         # NEW CODE: Skips Ollama if activation found anything
#         # ========================================
#         skip_ollama = False
#         ollama_output = {}
        
#         # Process activation detections (O(k) where k = detections, typically 1-3)
#         for detected_name in activation_detected:
#             if detected_name in activation_mappings:
#                 cid = activation_mappings[detected_name]
#                 ollama_output[cid] = "Y"
#                 skip_ollama = True  # KEY: This prevents Ollama from running
#                 logger.info(f"   → Mapped {detected_name} to class {cid}")

#         # ========================================
#         # STEP 3: Conditionally run Ollama (SLOW - ~5-10 seconds)
#         # ========================================
#         if skip_ollama:
#             activation_processed += 1
#             #aaalogger.info(f"   ⚡ FAST PATH: Skipped Ollama (saved ~8s)")
#         else:
#             #logger.info(f"   🐌 SLOW PATH: Running Ollama analysis...")
#             ollama_output = analyze_image(
#                 local_path, ollama_host, prompts, class_ids, model_name
#             )
#             ollama_processed += 1
#             logger.info(f"   ✓ Ollama completed")

#         # Handle visicooler detection data if present
#         if ollama_output.get("1001") == "Y":
#             cur.execute("""
#                 SELECT numshelf, numpureshelf, percentrgb, coolersize,
#                        chilleditems, warmitems, skus_detected,
#                        share_chilled, share_warm, present_no_facings
#                 FROM orgi.visibilitydetails
#                 WHERE cyclecountid=%s AND imagefilename=%s
#             """, (cyclecountid, filename))
#             r = cur.fetchone()
#             if r:
#                 ollama_output.update({
#                     "1003": r[3], "1012": str(r[0]), "1013": str(r[1]),
#                     "1014": str(r[2]), "1047": r[6], "1048": str(r[4]),
#                     "1049": str(r[5]), "1050": str(r[7]),
#                     "1051": str(r[8]), "1052": r[9]
#                 })

#         now = datetime.now()
#         s3_annot = f"{s3_annotated_folder}/{filename}"

#         for cid, val in ollama_output.items():
#             if cid not in class_ids:
#                 continue
#             inference = 1.0
#             if int(cid) in [1012,1013,1014,1048,1049,1050,1051]:
#                 inference = float(val) if str(val).replace(".", "").isdigit() else 0.0

#             results.append({
#                 "rowid": rowid,
#                 "modelname": model_name,
#                 "imagefilename": filename,
#                 "classid": int(cid),
#                 "classtext": get_classtext(cur, int(cid)),
#                 "value": val,
#                 "inference": inference,
#                 "modelrun": now,
#                 "processed_flag": "N",
#                 "storeid": storeid,
#                 "storename": storename,
#                 "s3path_actual_file": s3_key,
#                 "s3path_annotated_file": s3_annot
#             })
#             rowid += 1

#         s3_handler.upload_file_to_s3(local_path, s3_annot)

#     # Log statistics
#     logger.info("=" * 60)
#     logger.info("PROCESSING STATISTICS:")
#     logger.info(f"Total images processed: {total_images}")
#     logger.info(f"Handled by activation model: {activation_processed}")
#     logger.info(f"Handled by Ollama: {ollama_processed}")
#     logger.info(f"Time saved by activation model: ~{activation_processed} Ollama calls skipped")
#     logger.info("=" * 60)

#     if results:
#         os.makedirs(os.path.dirname(output_csv), exist_ok=True)
#         with open(output_csv, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=results[0].keys())
#             writer.writeheader()
#             writer.writerows(results)

#     insert_ollama_results(cur, stagingid, results, model_name, s3_annotated_folder, image_paths)
#     conn.commit()
#     close_db_connection(conn, cur)

#     return results, output_csv




import os
import json
import csv
import logging
import re
import requests
from datetime import datetime
from ollama import Client
from ultralytics import YOLO

from app.config_loader import load_config, load_json_classes
from app.db_handler import (
    initialize_db_connection,
    close_db_connection,
    get_classtext,
    insert_ollama_results,
    get_max_stagingid
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_ollama_server(ollama_host, model_name):
    try:
        r = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        models = r.json().get("models", [])
        return any(m["name"] == model_name for m in models)
    except Exception as e:
        logger.error(f"Ollama server check failed: {e}")
        return False


def extract_json(text):
    if not text:
        return {}
    txt = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def ollama_generate(ollama_host, model_name, prompt, image_path):
    client = Client(host=ollama_host)
    try:
        r = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "Return ONLY a JSON object."},
                {"role": "user", "content": prompt, "images": [image_path]},
            ],
            format="json",
            options={"temperature": 0}
        )
        return extract_json(r["message"]["content"])
    except Exception as e:
        logger.warning(f"Ollama retry: {e}")
        try:
            r = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Return ONLY a JSON object."},
                    {"role": "user", "content": prompt, "images": [image_path]},
                ],
                options={"temperature": 0}
            )
            return extract_json(r["message"]["content"])
        except Exception:
            return {}

_ACTIVATION_YOLO = None

def get_activation_yolo(model_path):
    global _ACTIVATION_YOLO
    if _ACTIVATION_YOLO is None:
        logger.info(f"Loading activation YOLO: {model_path}")
        _ACTIVATION_YOLO = YOLO(model_path)
    return _ACTIVATION_YOLO


def run_activation_yolo(image_path, model_path, conf_threshold=0.3):
    """
    Run activation YOLO model with configurable confidence threshold.
    Returns set of detected class names (lowercased, underscored).
    """
    model = get_activation_yolo(model_path)
    results = model(image_path, conf=conf_threshold, verbose=False)

    detected = set()
    for r in results:
        for cls in r.boxes.cls:
            class_name = model.names[int(cls)].lower().replace(" ", "_")
            detected.add(class_name)
            logger.info(f"Activation YOLO detected: {class_name} in {os.path.basename(image_path)}")
    
    return detected

def analyze_image(image_path, ollama_host, prompts, class_ids, model_name):
    """
    Analyze image for visibility items ONLY (extended groups 1 and 2).
    
    IMPORTANT SCOPE:
    - This function ONLY analyzes class IDs 1018-1046 (promotional/merchandising items)
    - Visicooler detection (1001-1014) is handled by mobile app - NOT analyzed here
    - SKU/facing data (1047-1052) is handled by visicooler analysis - NOT analyzed here
    
    Mobile app stores visicooler data in orgi.visibilitydetails
    Visicooler module stores cooler metrics in orgi.coolermetricstransaction
    This function focuses solely on extended visibility items from prompt files.
    """
    merged = {}

    # Only analyze extended visibility items (1018-1046)
    # No visicooler detection or parameter extraction
    for p in ["extended_visibility_group1", "extended_visibility_group2"]:
        if p in prompts and prompts[p]:
            result = ollama_generate(ollama_host, model_name, prompts[p], image_path)
            merged.update(result)

    # Return only the class IDs that are in our scope (1018-1046)
    return {k: v for k, v in merged.items() if k in class_ids}

def run_ollama_analysis(
    image_paths,
    image_folder,
    output_csv,
    config_path,
    class_ids_path,
    ollama_host,
    s3_handler,
    s3_annotated_folder,
    db_config,
    cyclecountid,
    conn=None,
    cur=None
):
    config = load_config(config_path)
    ollama_cfg = config["ollama_config"]
    model_name = ollama_cfg["ollama_model"]
    prompts = ollama_cfg["prompts"]
    activation_yolo_model = ollama_cfg.get("activation_yolo_model")
    
    # Configurable confidence threshold (default 0.3)
    activation_conf_threshold = ollama_cfg.get("activation_conf_threshold", 0.3)

    if not check_ollama_server(ollama_host, model_name):
        logger.error("Ollama not available")
        return [], None

    class_ids = load_json_classes(class_ids_path)

    # Use existing connection if provided, otherwise create new one
    should_close_conn = False
    if conn is None or cur is None:
        conn, cur = initialize_db_connection(db_config)
        should_close_conn = True
    
    stagingid = get_max_stagingid(cur) + 1

    results = []
    rowid = 1
    
    # Stats tracking
    total_images = 0
    activation_processed = 0
    ollama_processed = 0

    # ========================================
    # KEY OPTIMIZATION: Map class IDs once (O(1) lookups)
    # ========================================
    activation_mappings = {
        "poster": "1019",
        "dps": "1053",
        "menu_board": "1023"
    }

    for (_, storename, filename, local_path, s3_key, storeid, subcategory_id) in image_paths:
        # Skip visicooler-specific subcategories (handled by visicooler module)
        if subcategory_id in [601, 602, 603, 604, 605]:
            continue
        if not os.path.exists(local_path):
            continue

        total_images += 1
        logger.info(f"[{total_images}] Processing: {filename}")
        
        # ========================================
        # STEP 1: Run activation YOLO (FAST - ~50ms)
        # ========================================
        activation_detected = set()
        if activation_yolo_model:
            activation_detected = run_activation_yolo(
                local_path, 
                activation_yolo_model, 
                conf_threshold=activation_conf_threshold
            )
            if activation_detected:
                logger.info(f"   ✓ Activation YOLO detected: {activation_detected}")
            else:
                logger.info(f"   ✗ Activation YOLO: No detections")

        # ========================================
        # STEP 2: Check if activation model can handle this image
        # ========================================
        skip_ollama = False
        ollama_output = {}
        
        # Process activation detections (only for class IDs 1018-1046)
        for detected_name in activation_detected:
            if detected_name in activation_mappings:
                cid = activation_mappings[detected_name]
                ollama_output[cid] = "Y"
                skip_ollama = True
                logger.info(f"   → Mapped {detected_name} to class {cid}")

        # ========================================
        # STEP 3: Conditionally run Ollama (SLOW - ~5-10 seconds)
        # SCOPE: Only analyzes extended visibility items (1018-1046)
        # Does NOT analyze visicooler or SKU data
        # ========================================
        if skip_ollama:
            activation_processed += 1
            logger.info(f"   ⚡ FAST PATH: Skipped Ollama (saved ~8s)")
        else:
            ollama_output = analyze_image(
                local_path, ollama_host, prompts, class_ids, model_name
            )
            ollama_processed += 1
            logger.info(f"   ✓ Ollama completed")

        # ========================================
        # IMPORTANT: NO DATABASE QUERY FOR VISICOOLER DATA
        # Removed the orgi.visibilitydetails query
        # Ollama ONLY handles class IDs from extended_group1.txt and extended_group2.txt
        # which are: 1018-1046 (promotional/merchandising items)
        # 
        # Visicooler data (1001-1014) - handled by mobile app
        # SKU/facing data (1047-1052) - handled by visicooler analysis module
        # ========================================

        now = datetime.now()
        s3_annot = f"{s3_annotated_folder}/{filename}"

        # Process only the output from Ollama (class IDs 1018-1046)
        for cid, val in ollama_output.items():
            if cid not in class_ids:
                continue
            
            # All extended visibility items have inference = 1.0 (binary Y/N/N/A detection)
            inference = 1.0

            results.append({
                "rowid": rowid,
                "modelname": model_name,
                "imagefilename": filename,
                "classid": int(cid),
                "classtext": get_classtext(cur, int(cid)),
                "value": val,
                "inference": inference,
                "modelrun": now,
                "processed_flag": "N",
                "storeid": storeid,
                "storename": storename,
                "s3path_actual_file": s3_key,
                "s3path_annotated_file": s3_annot
            })
            rowid += 1

        s3_handler.upload_file_to_s3(local_path, s3_annot)

    # Log statistics
    logger.info("=" * 60)
    logger.info("OLLAMA ANALYSIS STATISTICS:")
    logger.info(f"Total images processed: {total_images}")
    logger.info(f"Handled by activation model: {activation_processed}")
    logger.info(f"Handled by Ollama: {ollama_processed}")
    logger.info(f"Time saved: ~{activation_processed * 8}s ({activation_processed} Ollama calls skipped)")
    logger.info(f"Scope: Extended visibility items only (class IDs 1018-1046)")
    logger.info("=" * 60)

    if results:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    insert_ollama_results(cur, stagingid, results, model_name, s3_annotated_folder, image_paths)
    conn.commit()
    
    # Only close connection if we created it
    if should_close_conn:
        close_db_connection(conn, cur)

    return results, output_csv