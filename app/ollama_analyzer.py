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


def run_activation_yolo(image_path, model_path):
    model = get_activation_yolo(model_path)
    results = model(image_path, conf=0.3, verbose=False)

    detected = set()
    for r in results:
        for cls in r.boxes.cls:
            detected.add(
                model.names[int(cls)].lower().replace(" ", "_")
            )
    return detected

def analyze_image(image_path, ollama_host, prompts, class_ids, model_name):
    merged = {}

    detect = ollama_generate(
        ollama_host, model_name, prompts["visicooler_detect"], image_path
    )
    merged.update({str(k): v for k, v in detect.items()})

    if merged.get("1001") == "N":
        for k in range(1002, 1012):
            merged[str(k)] = "N/A"
    else:
        attrs = ollama_generate(
            ollama_host, model_name, prompts["visicooler_attrs"], image_path
        )
        merged.update({str(k): v for k, v in attrs.items()})

        for p in ["extended_visibility_group1", "extended_visibility_group2"]:
            merged.update(
                ollama_generate(ollama_host, model_name, prompts[p], image_path)
            )

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
    cyclecountid
):
    config = load_config(config_path)
    ollama_cfg = config["ollama_config"]
    model_name = ollama_cfg["ollama_model"]
    prompts = ollama_cfg["prompts"]
    activation_yolo_model = ollama_cfg.get("activation_yolo_model")

    if not check_ollama_server(ollama_host, model_name):
        logger.error("Ollama not available")
        return [], None

    class_ids = load_json_classes(class_ids_path)

    conn, cur = initialize_db_connection(db_config)
    stagingid = get_max_stagingid(cur) + 1

    results = []
    rowid = 1

    for (_, storename, filename, local_path, s3_key, storeid, subcategory_id) in image_paths:
        if subcategory_id in [601, 602, 603, 604, 605]:
            continue
        if not os.path.exists(local_path):
            continue

        # Run activation YOLO to check for poster, dps, menu_board
        activation_detected = set()
        if activation_yolo_model:
            activation_detected = run_activation_yolo(local_path, activation_yolo_model)

        # Map detected classes to specific class IDs
        activation_mappings = {
            "Poster": "1019",
            "DPS": "1053",
            "Menu Board": "1023"
        }

        skip_ollama = False
        ollama_output = {}
        for detected_name in activation_detected:
            if detected_name in activation_mappings:
                cid = activation_mappings[detected_name]
                ollama_output[cid] = "Y"
                skip_ollama = True

        if not skip_ollama:
            # Run full Ollama analysis
            ollama_output = analyze_image(
                local_path, ollama_host, prompts, class_ids, model_name
            )

        # Additional logic for yolo_hits (existing code, but may be redundant now)
        # if not skip_ollama:
        #     for yolo_name in activation_detected:
        #         for cid in class_ids:
        #             if yolo_name in get_classtext(cur, int(cid)).lower():
        #                 ollama_output[str(cid)] = "Y"

        if ollama_output.get("1001") == "Y":
            cur.execute("""
                SELECT numshelf, numpureshelf, percentrgb, coolersize,
                       chilleditems, warmitems, skus_detected,
                       share_chilled, share_warm, present_no_facings
                FROM orgi.visibilitydetails
                WHERE cyclecountid=%s AND imagefilename=%s
            """, (cyclecountid, filename))
            r = cur.fetchone()
            if r:
                ollama_output.update({
                    "1003": r[3], "1012": str(r[0]), "1013": str(r[1]),
                    "1014": str(r[2]), "1047": r[6], "1048": str(r[4]),
                    "1049": str(r[5]), "1050": str(r[7]),
                    "1051": str(r[8]), "1052": r[9]
                })

        now = datetime.now()
        s3_annot = f"{s3_annotated_folder}/{filename}"

        for cid, val in ollama_output.items():
            if cid not in class_ids:
                continue
            inference = 1.0
            if int(cid) in [1012,1013,1014,1048,1049,1050,1051]:
                inference = float(val) if str(val).replace(".", "").isdigit() else 0.0

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

    if results:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    insert_ollama_results(cur, stagingid, results, model_name, s3_annotated_folder, image_paths)
    conn.commit()
    close_db_connection(conn, cur)

    return results, output_csv
