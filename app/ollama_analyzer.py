import os
import json
import csv
import logging
import re
from datetime import datetime
from ollama import Client
import requests
from app.config_loader import load_config, load_json_classes
from app.db_handler import initialize_db_connection, close_db_connection, get_classtext, insert_ollama_results, get_max_stagingid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_ollama_server(ollama_host, model_name="qwen2.5vl:latest"):
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

def extract_json(text):
    """Extract JSON object from text, removing code fences and handling encoding issues."""
    if not text:
        return {}
    txt = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return {}
    snippet = m.group(0).strip()
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        snippet2 = snippet.replace("\ufeff", "").replace("\u200b", "")
        try:
            return json.loads(snippet2)
        except Exception:
            return {}

def ollama_generate(ollama_host, model_name, prompt, image_path):
    """Generate response from Ollama with retry logic."""
    client = Client(host=ollama_host)
    try:
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "Return ONLY a JSON object. No code fences, no extra text."},
                {"role": "user", "content": prompt, "images": [image_path]},
            ],
            format="json",
            options={"temperature": 0}
        )
        content = response.get('message', {}).get('content', "")
        try:
            return json.loads(content) if isinstance(content, str) else content
        except Exception:
            return extract_json(content)
    except Exception as e:
        logger.warning(f"First attempt failed: {e}")
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "Return ONLY a JSON object. No code fences, no extra text."},
                {"role": "user", "content": prompt, "images": [image_path]},
            ],
            options={"temperature": 0}
        )
        content = response.get('message', {}).get('content', "")
        return extract_json(content)

def analyze_image(image_path, ollama_host, prompts, class_ids, model_name="qwen2.5vl:latest", cyclecountid=None, imagefilename=None, cur=None, hybrid_mode=True):
    """Analyze image with Ollama using qwen2.5vl:latest model."""
    merged = {}
    detection_prompt = prompts.get("visicooler_detect", "")
    detection_result = ollama_generate(ollama_host, model_name, detection_prompt, image_path)
    if isinstance(detection_result, dict):
        merged.update({str(k): v for k, v in detection_result.items()})
    if merged.get("1001") == "N":
        for k in range(1002, 1012):
            merged[str(k)] = "N/A"
    else:
        parameters_prompt = prompts.get("visicooler_attrs", "")
        if parameters_prompt:
            params_result = ollama_generate(ollama_host, model_name, parameters_prompt, image_path)
            if isinstance(params_result, dict):
                merged.update({str(k): v for k, v in params_result.items()})
    for pfile in ["extended_visibility_group1", "extended_visibility_group2"]:
        prompt = prompts.get(pfile, "")
        if prompt:
            result = ollama_generate(ollama_host, model_name, prompt, image_path)
            if isinstance(result, dict):
                merged.update({str(k): v for k, v in result.items()})
    return {k: v for k, v in merged.items() if k in class_ids}

def run_ollama_analysis(image_paths, image_folder, output_csv, config_path, class_ids_path, ollama_host, s3_handler, s3_annotated_folder, db_config, cyclecountid):
    """Run Ollama analysis on images and save results to CSV and database."""
    try:
        # Load configuration
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        ollama_enabled = config.get('ollama_config', {}).get('ollama_enabled', True)
        model_name = config.get('ollama_config', {}).get('ollama_model', 'qwen2.5vl:latest')
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
        for idx, (filesequenceid, storename, filename, local_path, s3_key, storeid, subcategory_id) in enumerate(image_paths):
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
                        SELECT numshelf, numpureshelf, percentrgb, coolersize, chilleditems, warmitems, skus_detected,
                               share_chilled, share_warm, present_no_facings
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
                        "1049": str(db_result[5]) if db_result and db_result[5] is not None else "0",
                        "1050": str(db_result[7]) if db_result and db_result[7] is not None else "0.0",
                        "1051": str(db_result[8]) if db_result and db_result[8] is not None else "0.0",
                        "1052": db_result[9] if db_result and db_result[9] is not None else "N"
                    }
                else:
                    db_results = {
                        "1003": "Unknown",
                        "1012": "0",
                        "1013": "0",
                        "1014": "0.0",
                        "1047": "N",
                        "1048": "0",
                        "1049": "0",
                        "1050": "0.0",
                        "1051": "0.0",
                        "1052": "N"
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
                        if classid_int in [1012, 1013, 1014, 1048, 1049, 1050, 1051]:
                            inference = float(value) if str(value).replace('.', '').replace('%', '').isdigit() else 0.0
                        elif classid_int in [1047, 1052]:
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
            logger.warning("No valid Ollama results to save to CSV or database.")

        close_db_connection(conn, cur)
        return results, output_csv
    except Exception as e:
        logger.error(f"Error in Ollama analysis: {e}")
        if 'conn' in locals():
            close_db_connection(conn, cur)
        raise
