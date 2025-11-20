import os
import json
import csv
import logging
from datetime import datetime
from ollama import Client
from PIL import Image
import base64
from io import BytesIO
import requests
from app.config_loader import load_config, load_json_classes
from app.db_handler import initialize_db_connection, close_db_connection, get_classtext, insert_ollama_results, get_max_stagingid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_ollama_server(ollama_host, model_name="llava:7b"):
    """Check if the Ollama server is reachable and the specified model is available."""
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if any(model['name'] == model_name for model in models):
                logger.info(f"Ollama server at {ollama_host} is reachable and '{model_name}' is available.")
                return True
            else:
                logger.error(
                    f"Ollama server at {ollama_host} is reachable, but '{model_name}' is not available. "
                    f"Available models: {[m['name'] for m in models]}. "
                    f"To install '{model_name}', run: `ollama pull {model_name}` on the server."
                )
                return False
        else:
            logger.error(f"Ollama server at {ollama_host} returned status code {response.status_code}.")
            return False
    except Exception as e:
        logger.error(
            f"Failed to connect to Ollama server at {ollama_host}: {e}. "
            f"Ensure the server is running and has sufficient resources for '{model_name}'."
        )
        return False

def encode_image(image_path):
    """Encode image to base64 string with optimized quality."""
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=90)  # Reduced quality to save memory
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None

def default_visicooler_output():
    """Return default visicooler output."""
    return {
        "1001": "N",
        "1002": "Unknown",
        "1004": "N/A",
        "1005": "N/A",
        "1006": "N/A",
        "1007": "N/A",
        "1008": "N/A",
        "1009": "N/A",
        "1010": "N/A",
        "1011": "N/A",
    }

def default_extended_visibility(class_ids):
    """Return default extended visibility output."""
    extended_ids = [str(cid) for cid in range(1018, 1047) if str(cid) in class_ids]
    return {cid: "N" for cid in extended_ids}

def normalize_visicooler_attrs(raw, class_ids):
    """Normalize visicooler attributes with strict validation."""
    out = default_visicooler_output()
    if not isinstance(raw, dict):
        logger.warning("Invalid visicooler response: not a dict. Using defaults.")
        return {k: v for k, v in out.items() if k in class_ids}
    
    v1001 = str(raw.get("1001", "N")).upper().strip()
    v1001 = "Y" if v1001 == "Y" else "N"
    out["1001"] = v1001
    
    brand = str(raw.get("1002", "Unknown")).strip()

    # Normalize common brand variations
    brand_lower = brand.lower()
    if "coca" in brand_lower or "coke" in brand_lower:
        brand = "Coca-Cola"
    elif "pepsi" in brand_lower:
        brand = "Pepsi"
    elif "unbranded" in brand_lower or "generic" in brand_lower:
        brand = "Unbranded"
    else:
        brand = "Unknown"
    out["1002"] = brand

    # Coca-Cola specific validations
    coke_only_allowed = {
        "1004": ["Y", "N", "N/A"],
        "1005": ["Y", "N", "N/A"],
        "1006": ["Y", "N", "N/A"],
        "1007": ["Fully", "Partial", "Not visible", "N/A"],
        "1008": ["Y", "N", "N/A"],
        "1009": ["Y", "N", "N/A"],
        "1010": ["Y", "N", "N/A"],
        "1011": ["100% pure", "Impure", "N/A"],
    }
    if brand == "Coca-Cola" and v1001 == "Y":
        for k, allowed in coke_only_allowed.items():
            val = raw.get(k, "N/A")
            if isinstance(val, str):
                val = val.strip()
            if k == "1007" and isinstance(val, str):
                low = val.lower()
                if "full" in low:
                    val = "Fully"
                elif "part" in low:
                    val = "Partial"
                elif "not" in low or "hidden" in low:
                    val = "Not visible"
                else:
                    val = "N/A"
            elif k == "1011" and isinstance(val, str):
                low = val.lower()
                if "pure" in low and "100" in low:
                    val = "100% pure"
                elif "impure" in low or "mixed" in low:
                    val = "Impure"
                else:
                    val = "N/A"
            if val not in allowed:
                logger.warning(f"Invalid value {val} for class {k}. Defaulting to 'N/A'.")
                val = "N/A"
            out[k] = val
    else:
        for k in coke_only_allowed.keys():
            out[k] = "N/A"
    if v1001 == "N":
        out.update(default_visicooler_output())
    return {k: v for k, v in out.items() if k in class_ids}

def normalize_extended_visibility(raw, class_ids, visicooler_detected=False, is_indoor_only=False):
    """Normalize extended visibility attributes with stricter context-aware rules."""
    extended_ids = [str(cid) for cid in range(1018, 1047) if str(cid) in class_ids]
    out = {cid: "N" for cid in extended_ids}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k in out:
                val_str = str(v).strip().upper()
                if val_str in ["Y", "N", "N/A"]:
                    out[k] = val_str
                else:
                    logger.warning(f"Invalid value {val_str} for class {k}. Defaulting to 'N'.")
                    out[k] = "N"
    
    # Stricter post-processing rules for accuracy
    outdoor_items = ["1032", "1033", "1041"]
    if is_indoor_only:
        for item in outdoor_items:
            if item in out:
                out[item] = "N/A"
                logger.debug(f"Set class {item} to 'N/A' due to indoor-only scene.")
    if not visicooler_detected and "1039" in out:
        out["1039"] = "N/A"
        logger.debug("Set class 1039 to 'N/A' due to no visicooler detected.")
    
    return out

def ollama_generate(ollama_host, model_name, prompt, image_data):
    """Generate response from Ollama with retry logic and memory error handling."""
    client = Client(host=ollama_host)
    for attempt in range(3):
        try:
            response = client.generate(model=model_name, prompt=prompt, images=[image_data], format='json')
            result = json.loads(response.get('response', '{}'))
            logger.debug(f"Ollama response for prompt: {result}")
            return result
        except Exception as e:
            logger.warning(f"Ollama generate attempt {attempt+1} failed: {e}")
            if "memory" in str(e).lower():
                logger.error(f"Memory error detected: {e}. Using default output.")
                return {}
            if attempt == 2:
                logger.error("Ollama generate failed after retries.")
                return {}
    return {}

def fetch_yolo_bboxes(cur, cyclecountid, imagefilename):
    """Fetch YOLO bounding boxes from orgi.cyclecount_staging."""
    try:
        cur.execute(
            """
            SELECT classid, x1, x2, y1, y2
            FROM orgi.cyclecount_staging
            WHERE cyclecountid = %s AND imagefilename = %s
            """,
            (cyclecountid, imagefilename)
        )
        bboxes = cur.fetchall()
        logger.debug(f"Fetched {len(bboxes)} YOLO bboxes for {imagefilename}")
        return bboxes
    except Exception as e:
        logger.error(f"Failed to fetch YOLO bboxes for {imagefilename}: {e}")
        return []

def analyze_image(image_path, ollama_host, prompts, class_ids, model_name="llava:7b", cyclecountid=None, imagefilename=None, cur=None, hybrid_mode=True):
    """Analyze image with Ollama, incorporating hybrid YOLO approach."""
    encoded_image = encode_image(image_path)
    if not encoded_image:
        logger.warning(f"Failed to encode {image_path}, using defaults.")
        return default_visibility(class_ids)

    # Scene type detection (indoor-only)
    scene_prompt = "Is this image indoor-only (no windows/exterior visible)? Respond with JSON: {'indoor': 'Y/N'}"
    scene_result = ollama_generate(ollama_host, model_name, scene_prompt, encoded_image)
    if not scene_result:
        logger.warning("Scene detection failed, defaulting to indoor='N'.")
        is_indoor_only = False
    else:
        is_indoor_only = scene_result.get('indoor', 'N') == 'Y'
    logger.debug(f"Scene detection: indoor_only={is_indoor_only}")

    # Visicooler detection
    detection = ollama_generate(ollama_host, model_name, prompts['visicooler_detect'], encoded_image)
    if not detection:
        logger.warning("Visicooler detection failed, defaulting to 'N'.")
        det_value = "N"
    else:
        det_value = str(detection.get("1001", "N")).upper()
        det_value = "Y" if det_value == "Y" else "N"
    logger.debug(f"Visicooler detection: {det_value}")

    # Visicooler attributes
    if det_value == "Y":
        attrs = ollama_generate(ollama_host, model_name, prompts['visicooler_attrs'], encoded_image)
        if not attrs:
            logger.warning("Visicooler attributes analysis failed, using defaults.")
            visicooler_out = default_visicooler_output()
            visicooler_out["1001"] = "Y"
        else:
            visicooler_out = normalize_visicooler_attrs(attrs, class_ids)
            visicooler_out["1001"] = "Y"
    else:
        visicooler_out = default_visicooler_output()
        visicooler_out["1001"] = "N"

    # Extended visibility (split prompts)
    ext_group1 = ollama_generate(ollama_host, model_name, prompts.get('extended_visibility_group1', prompts['extended_visibility']), encoded_image)
    ext_group2 = ollama_generate(ollama_host, model_name, prompts.get('extended_visibility_group2', prompts['extended_visibility']), encoded_image)
    if not ext_group1 and not ext_group2:
        logger.warning("Extended visibility analysis failed, using defaults.")
        ext = {}
    else:
        ext = {**ext_group1, **ext_group2}
    logger.debug(f"Extended visibility raw output: {ext}")

    # Hybrid approach: Crop and re-analyze for specific classes
    if hybrid_mode and cur and cyclecountid and imagefilename:
        bboxes = fetch_yolo_bboxes(cur, cyclecountid, imagefilename)
        img = Image.open(image_path).convert('RGB')
        for classid, x1, x2, y1, y2 in bboxes:
            if str(classid) in class_ids and str(classid) in ["1042", "1043", "1044", "1045", "1046"]:
                try:
                    crop = img.crop((x1, y1, x2, y2))
                    crop_buffer = BytesIO()
                    crop.save(crop_buffer, format="JPEG", quality=90)
                    crop_encoded = base64.b64encode(crop_buffer.getvalue()).decode('utf-8')
                    crop_prompt = f"Analyze this cropped image for Coca-Cola branded item class {classid}. Include any text or logos visible: Respond with JSON: {{'{classid}': 'Y/N/N/A'}}"
                    crop_result = ollama_generate(ollama_host, model_name, crop_prompt, crop_encoded)
                    if crop_result.get(str(classid)) in ["Y", "N", "N/A"]:
                        ext[str(classid)] = crop_result[str(classid)]
                        logger.debug(f"Hybrid mode updated class {classid} to {crop_result[str(classid)]}")
                except Exception as e:
                    logger.warning(f"Failed to process crop for class {classid}: {e}")

    ext_out = normalize_extended_visibility(ext, class_ids, visicooler_detected=(det_value == "Y"), is_indoor_only=is_indoor_only)
    return {**visicooler_out, **ext_out}

def default_visibility(class_ids):
    """Return default visibility results."""
    visicooler_out = default_visicooler_output()
    ext_out = default_extended_visibility(class_ids)
    return {**visicooler_out, **ext_out}

def run_ollama_analysis(image_paths, image_folder, output_csv, config_path, class_ids_path, ollama_host, s3_handler, s3_annotated_folder, db_config, cyclecountid):
    """Run Ollama analysis on images and save results to CSV and database."""
    try:
        # Load configuration
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        ollama_enabled = config.get('ollama_config', {}).get('ollama_enabled', True)
        model_name = config.get('ollama_config', {}).get('ollama_model', 'llava:7b')
        prompts = config.get('ollama_config', {}).get('prompts', {})
        hybrid_mode = config.get('ollama_config', {}).get('hybrid_mode', True)

        if not ollama_enabled:
            logger.warning("Ollama analysis is disabled in config. Skipping analysis.")
            return [], None

        if not all(prompts.values()):
            logger.error("One or more prompt contents are missing or empty.")
            raise ValueError("One or more prompt contents are missing or empty.")

        # Load class IDs
        class_ids = load_json_classes(class_ids_path)
        logger.info(f"Loaded class IDs: {class_ids}")

        # Check Ollama server
        if not check_ollama_server(ollama_host, model_name):
            logger.error(
                f"Skipping Ollama analysis due to unavailable '{model_name}' model. "
                f"To enable, install the model using: `ollama pull {model_name}`."
            )
            return [], None

        # Initialize database connection
        conn, cur = initialize_db_connection(db_config)
        stagingid = get_max_stagingid(cur) + 1
        logger.info(f"Using stagingid: {stagingid}")
        results = []

        # Get max rowid for this stagingid
        cur.execute("SELECT MAX(rowid) FROM orgi.visibilityitemsstaging WHERE stagingid = %s", (stagingid,))
        result = cur.fetchone()
        rowid_counter = (result[0] if result[0] is not None else 0) + 1

        # Ensure S3 folder aligns with stagingid
        s3_annotated_folder = f"ModelResults/VisibleItem_{stagingid}"

        # Process images
        for idx, (filesequenceid, storename, filename, local_path, s3_key, storeid) in enumerate(image_paths):
            logger.info(f"Processing image {idx + 1}/{len(image_paths)}: {filename}")
            try:
                # Validate image file
                if not os.path.exists(local_path):
                    logger.error(f"Image file not found: {local_path}")
                    continue

                # Analyze image with Ollama
                ollama_output = analyze_image(local_path, ollama_host, prompts, class_ids, model_name, cyclecountid, filename, cur, hybrid_mode)
                logger.debug(f"Ollama output for {filename}: {ollama_output}")

                # Fetch visibility details from orgi.visibilitydetails only if visicooler detected
                if ollama_output.get("1001") == "Y":
                    cur.execute(
                        """
                        SELECT numshelf, numpureshelf, percentrgb, coolersize, chilleditems, warmitems, skus_detected
                        FROM orgi.visibilitydetails
                        WHERE cyclecountid = %s AND imagefilename = %s
                        """,
                        (cyclecountid, filename)
                    )
                    db_result = cur.fetchone()
                    db_results = {
                        "1003": db_result[3] if db_result and db_result[3] is not None else "Unknown",
                        "1012": str(db_result[0]) if db_result and db_result[0] is not None else "0",
                        "1013": str(db_result[1]) if db_result and db_result[1] is not None else "0",
                        "1014": str(db_result[2]) if db_result and db_result[2] is not None else "0.0",
                        "1047": db_result[6] if db_result and db_result[6] is not None else "N",
                        "1048": str(db_result[4]) if db_result and db_result[4] is not None else "0",
                        "1049": str(db_result[5]) if db_result and db_result[5] is not None else "0"
                    }
                else:
                    db_results = {
                        "1003": "Unknown",
                        "1012": "0",
                        "1013": "0",
                        "1014": "0.0",
                        "1047": "N",
                        "1048": "0",
                        "1049": "0"
                    }

                # Combine results
                combined_results = {**ollama_output, **db_results}
                now = datetime.now()
                s3_annotated_key = f"{s3_annotated_folder}/{filename}"

                for classid_str, value in combined_results.items():
                    try:
                        classid_int = int(classid_str)
                        if str(classid_int) not in class_ids:
                            continue
                        classtext = get_classtext(cur, classid_int)
                        if classid_int in [1012, 1013, 1014, 1048, 1049]:
                            inference = float(value) if str(value).replace('.', '').replace('%', '').isdigit() else 0.0
                        elif classid_int == 1047:
                            inference = 1.0 if value in ['Y', 'N'] else 0.0
                        else:
                            inference = 1.0
                        results.append({
                            'rowid': rowid_counter,
                            'modelname': model_name,
                            'imagefilename': filename,
                            'classid': classid_int,
                            'classtext': classtext,
                            'value': str(value),
                            'inference': inference,
                            'modelrun': now,
                            'processed_flag': 'N',
                            'storeid': storeid,
                            'storename': storename,
                            's3path_actual_file': s3_key,
                            's3path_annotated_file': s3_annotated_key
                        })
                        rowid_counter += 1
                    except ValueError:
                        logger.warning(f"Invalid classid {classid_str} for {filename}, skipping.")
                        continue

                # Upload images to S3
                s3_handler.upload_file_to_s3(local_path, s3_annotated_key)
                logger.info(f"Uploaded {local_path} to S3: {s3_annotated_folder}/{filename}")

            except Exception as e:
                logger.error(f"Error processing image {filename}: {e}")
                continue

        # Save results to CSV
        if results:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['rowid', 'modelname', 'imagefilename', 'classid', 'classtext', 'value', 'inference', 'modelrun', 'processed_flag', 'storeid', 'storename', 's3path_actual_file', 's3path_annotated_file']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    writer.writerow(result)
            logger.info(f"Results saved to {output_csv}")

            # Insert results into orgi.visibilityitemsstaging
            insert_ollama_results(cur, stagingid, results, model_name, s3_annotated_folder, image_paths)
            conn.commit()
            logger.info(f"Inserted {len(results)} rows into orgi.visibilityitemsstaging with stagingid {stagingid}")
        else:
            logger.warning("No valid Ollama results to save to CSV or database. Likely due to memory constraints.")

        close_db_connection(conn, cur)
        return results, output_csv
    except Exception as e:
        logger.error(f"Error in Ollama analysis: {e}")
        if 'conn' in locals():
            close_db_connection(conn, cur)
        raise