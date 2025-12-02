import matplotlib
matplotlib.use('Agg')
import traceback
import logging
import os
import tempfile
from app.config_loader import load_config
from app.s3_handler import S3Handler
from app.yolo_predictor import run_yolo_predictions
from app.ollama_analyzer import run_ollama_analysis
from app.uploader import upload_ollama_results
from app.file_uploader import FileUploader
from app.db_handler import initialize_db_connection, close_db_connection, get_max_cyclecountid
from app.visicooler import run_visicooler_analysis, check_visibilitydetails_schema


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    conn = None
    cur = None
    try:
        # Load configuration
        config = load_config('config.json')
        yolo_config = config['yolo_config']
        ollama_config = config['ollama_config']
        s3_config = config['s3_config']
        db_config = config['db_config']
        visicooler_config = config['visicooler_config']
        hybrid_mode = ollama_config.get('hybrid_mode', True)

        # Initialize database connection
        conn, cur = initialize_db_connection(db_config)

        # # Check visibilitydetails schema
        if not check_visibilitydetails_schema(cur):
            logger.error("Schema validation failed for orgi.visibilitydetails. Please verify the table schema using: \\d orgi.visibilitydetails. Ensure columns match: cyclecountid (INTEGER/BIGINT), imagefilename (VARCHAR/TEXT/CHARACTER VARYING), numshelf (INTEGER/BIGINT), numproducts (INTEGER/BIGINT), numpureshelf (INTEGER/BIGINT), coolersize (VARCHAR/TEXT/CHARACTER VARYING), percentrgb (FLOAT/NUMERIC), chilleditems (INTEGER/BIGINT), warmitems (INTEGER/BIGINT), skus_detected (VARCHAR/TEXT/CHARACTER VARYING), share_chilled (FLOAT/NUMERIC), share_warm (FLOAT/NUMERIC), present_no_facings (VARCHAR/TEXT/CHARACTER VARYING).")
            return

        # Initialize S3 handler
        s3_handler = S3Handler(s3_config, db_config)

        # Create temporary directory for image downloads
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Created temporary directory: {temp_dir}")

            # Download images from S3
            logger.info("Downloading images from S3...")
            image_paths, failed_files = s3_handler.download_images_from_s3(temp_dir)
            if not image_paths:
                logger.warning("No images downloaded successfully. Exiting.")
                return
            if failed_files:
                logger.info(f"Proceeding with {len(image_paths)} images, {len(failed_files)} failed.")

            # Get or increment cyclecountid
            # Compute cyclecountid reliably from DB maxima across key tables
            try:
                cur.execute("""
                    SELECT GREATEST(
                        COALESCE((SELECT MAX(cyclecountid) FROM orgi.visibilitydetails), 0),
                        COALESCE((SELECT MAX(cyclecountid) FROM orgi.cyclecount_staging), 0),
                        COALESCE((SELECT MAX(cyclecountid) FROM orgi.coolermetricsmaster), 0)
                    ) AS max_cycle
                """)
                row = cur.fetchone()
                max_cycle = int(row[0]) if row and row[0] is not None else 0
                cyclecountid = max_cycle + 1
                logger.info(f"Computed cyclecountid from DB maxima: {cyclecountid} (max seen: {max_cycle})")
            except Exception as e:
                logger.warning(f"Failed to compute cyclecountid from DB maxima, falling back to get_max_cyclecountid(): {e}")
                try:
                    cyclecountid = get_max_cyclecountid(cur) + 1
                    logger.info(f"Fallback cyclecountid: {cyclecountid}")
                except Exception as e2:
                    logger.error(f"Fallback get_max_cyclecountid failed: {e2}. Using cyclecountid = 1")
                    cyclecountid = 1


            # Run YOLO predictions
            logger.info("Running YOLO predictions...")
            yolo_cyclecountid, database_success = run_yolo_predictions(
                yaml_path=yolo_config['yaml_path'],
                model_path=yolo_config['model_path'],
                image_folder=temp_dir,
                csv_output_path=yolo_config['csv_output_path'],
                modelname=yolo_config['modelname'],
                s3_bucket_name=s3_config['bucket_name'],
                s3_folder=yolo_config['output_s3_folder'],
                conn=conn,
                cur=cur,
                s3_handler=s3_handler,
                image_paths=image_paths,
                cyclecountid_override=cyclecountid
            )
            if yolo_cyclecountid != cyclecountid:
                logger.warning(f"⚠ YOLO returned cyclecountid {yolo_cyclecountid}, "
                               f"but pipeline will use {cyclecountid}.")

            # # Run visicooler analysis
            logger.info("Running visicooler analysis...")
            try:
                visicooler_records = run_visicooler_analysis(
                    image_paths=image_paths,
                    config=config,
                    s3_handler=s3_handler,
                    conn=conn,
                    cur=cur,
                    output_folder_path=visicooler_config['output_folder_path'],
                    cyclecountid=cyclecountid
                )
                logger.info(f"Visicooler analysis completed: {len(visicooler_records)} records generated.")
            except Exception as e:
                logger.error(f"Visicooloer analysis failed: {e}. Continuing pipeline.")
                visicooler_records = []

            # Upload YOLO CSV to S3
            logger.info("Uploading YOLO CSV to S3...")
            try:
                s3_handler.upload_file_to_s3(
                    yolo_config['csv_output_path'],
                    f"{yolo_config['output_s3_folder']}CycleCount_{cyclecountid}/prediction_output.csv"
                )
            except Exception as e:
                logger.error(f"Failed to upload YOLO CSV to S3: {e}")

            # # Run Ollama analysis
            logger.info("Running Ollama analysis with ollama:qwen2.5vl...")
            try:
                ollama_results, ollama_csv = run_ollama_analysis(
                    image_paths=image_paths,
                    image_folder=temp_dir,
                    output_csv=ollama_config['output_csv'],
                    config_path='config.json',
                    class_ids_path=ollama_config['class_ids_path'],
                    ollama_host=ollama_config['ollama_host'],
                    s3_handler=s3_handler,
                    s3_annotated_folder=f"{ollama_config['output_s3_folder']}VisibleItem_{cyclecountid}",
                    db_config=db_config,
                    cyclecountid=cyclecountid
                )
            except Exception as e:
                logger.error(f"Ollama analysis failed: {e}. Likely due to memory constraints. Consider upgrading system memory or using a smaller model.")
                ollama_results, ollama_csv = [], None

            # Upload Ollama CSV to S3
            logger.info("Uploading Ollama CSV to S3...")
            if ollama_csv and os.path.exists(ollama_csv):
                try:
                    s3_handler.upload_file_to_s3(
                        ollama_csv,
                        f"{ollama_config['output_s3_folder']}VisibleItem_{cyclecountid}/analysis_results_one_hot.csv"
                    )
                except Exception as e:
                    logger.error(f"Failed to upload Ollama CSV to S3: {e}")
            else:
                logger.warning("Skipping Ollama CSV upload due to missing CSV file. Likely due to memory constraints.")

            # Update processed_flag in orgi.fileupload
            logger.info("Updating processed_flag for downloaded images...")
            try:
                file_uploader = FileUploader(None)
                failed_updates = file_uploader.update_processed_flag(conn, image_paths)
                if failed_updates:
                    logger.info(f"Failed to update processed_flag for {len(failed_updates)} files: {[f[2] for f in failed_updates]}")
                else:
                    logger.info(f"Successfully updated processed_flag for {len(image_paths)} files.")
            except Exception as e:
                logger.error(f"Error updating processed_flag: {e}")
                logger.info("Continuing pipeline despite processed_flag update failure.")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error(traceback.format_exc())
        if 'conn' in locals():
            conn.rollback()
    finally:
        if conn is not None:
            close_db_connection(conn, cur)
        logger.info("Pipeline execution completed.")

if __name__ == "__main__":
    main()
