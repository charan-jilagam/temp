# import matplotlib
# matplotlib.use('Agg')
# import traceback
# import logging
# import os
# import tempfile
# from app.config_loader import load_config
# from app.s3_handler import S3Handler
# from app.ollama_analyzer import run_ollama_analysis
# from app.file_uploader import FileUploader
# from app.db_handler import initialize_db_connection, close_db_connection
# from app.visicooler import run_visicooler_analysis, check_visibilitydetails_schema


# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('outputs/pipeline.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# def main():
#     conn = None
#     cur = None
#     try:
#         # Load configuration
#         config = load_config('config.json')
#         ollama_config = config['ollama_config']
#         s3_config = config['s3_config']
#         db_config = config['db_config']
#         visicooler_config = config['visicooler_config']

#         # Initialize database connection
#         conn, cur = initialize_db_connection(db_config)

#         # Check visibilitydetails schema
#         if not check_visibilitydetails_schema(cur):
#             logger.error("Schema validation failed for orgi.visibilitydetails.")
#             return

#         # Initialize S3 handler
#         s3_handler = S3Handler(s3_config, db_config)

#         # Create temporary directory for image downloads
#         with tempfile.TemporaryDirectory() as temp_dir:
#             logger.info(f"Created temporary directory: {temp_dir}")

#             # Download images from S3
#             logger.info("Downloading images from S3...")
#             image_paths, failed_files = s3_handler.download_images_from_s3(temp_dir)
#             if not image_paths:
#                 logger.warning("No images downloaded successfully. Exiting.")
#                 return
#             if failed_files:
#                 logger.info(f"Proceeding with {len(image_paths)} images, {len(failed_files)} failed.")

#             # Get cyclecountid for Ollama analysis
#             try:
#                 cur.execute("""
#                     SELECT GREATEST(
#                         COALESCE((SELECT MAX(cyclecountid) FROM orgi.visibilitydetails), 0),
#                         COALESCE((SELECT MAX(stagingid) FROM orgi.visibilityitemsstaging), 0)
#                     ) AS max_cycle
#                 """)
#                 row = cur.fetchone()
#                 max_cycle = int(row[0]) if row and row[0] is not None else 0
#                 cyclecountid = max_cycle + 1
#                 logger.info(f"Computed cyclecountid: {cyclecountid}")
#             except Exception as e:
#                 logger.error(f"Failed to compute cyclecountid: {e}. Using cyclecountid = 1")
#                 cyclecountid = 1

#             # Run visicooler analysis (handles subcategory 605 - cooler metrics)
#             logger.info("Running visicooler analysis for subcategory 605...")
#             try:
#                 visicooler_records = run_visicooler_analysis(
#                     image_paths=image_paths,
#                     config=config,
#                     s3_handler=s3_handler,
#                     conn=conn,
#                     cur=cur,
#                     output_folder_path=visicooler_config['output_folder_path'],
#                     cyclecountid=cyclecountid
#                 )
#                 logger.info(f"Visicooler analysis completed: {len(visicooler_records)} records generated.")
#             except Exception as e:
#                 logger.error(f"Visicooler analysis failed: {e}")
#                 visicooler_records = []

#             # Run Ollama analysis (handles other subcategories - visibility items)
#             logger.info("Running Ollama analysis for visibility items...")
#             try:
#                 ollama_results, ollama_csv = run_ollama_analysis(
#                     image_paths=image_paths,
#                     image_folder=temp_dir,
#                     output_csv=ollama_config['output_csv'],
#                     config_path='config.json',
#                     class_ids_path=ollama_config['class_ids_path'],
#                     ollama_host=ollama_config['ollama_host'],
#                     s3_handler=s3_handler,
#                     s3_annotated_folder=f"{ollama_config['output_s3_folder']}VisibleItem_{cyclecountid}",
#                     db_config=db_config,
#                     cyclecountid=cyclecountid
#                 )
#                 logger.info(f"Ollama analysis completed: {len(ollama_results)} records generated.")
#             except Exception as e:
#                 logger.error(f"Ollama analysis failed: {e}")
#                 ollama_results, ollama_csv = [], None

#             # Upload Ollama CSV to S3
#             if ollama_csv and os.path.exists(ollama_csv):
#                 logger.info("Uploading Ollama CSV to S3...")
#                 try:
#                     s3_handler.upload_file_to_s3(
#                         ollama_csv,
#                         f"{ollama_config['output_s3_folder']}VisibleItem_{cyclecountid}/analysis_results.csv"
#                     )
#                     logger.info("Ollama CSV uploaded successfully.")
#                 except Exception as e:
#                     logger.error(f"Failed to upload Ollama CSV to S3: {e}")
#             else:
#                 logger.warning("No Ollama CSV to upload.")

#             # Update processed_flag in orgi.fileupload
#             logger.info("Updating processed_flag for processed images...")
#             try:
#                 file_uploader = FileUploader(None)
#                 failed_updates = file_uploader.update_processed_flag(conn, image_paths)
#                 if failed_updates:
#                     logger.warning(f"Failed to update processed_flag for {len(failed_updates)} files")
#                 else:
#                     logger.info(f"Successfully updated processed_flag for {len(image_paths)} files.")
#             except Exception as e:
#                 logger.error(f"Error updating processed_flag: {e}")

#             logger.info("=" * 60)
#             logger.info("PIPELINE SUMMARY:")
#             logger.info(f"Images processed: {len(image_paths)}")
#             logger.info(f"Visicooler records: {len(visicooler_records)}")
#             logger.info(f"Ollama records: {len(ollama_results) if ollama_results else 0}")
#             logger.info(f"Failed files: {len(failed_files)}")
#             logger.info("=" * 60)

#     except Exception as e:
#         logger.error(f"Error in main execution: {e}")
#         logger.error(traceback.format_exc())
#         if conn and 'conn' in locals():
#             conn.rollback()
#     finally:
#         if conn is not None:
#             close_db_connection(conn, cur)
#         logger.info("Pipeline execution completed.")

# if __name__ == "__main__":
#     main()


import matplotlib
matplotlib.use('Agg')
import traceback
import logging
import os
import tempfile
from app.config_loader import load_config
from app.s3_handler import S3Handler
from app.ollama_analyzer import run_ollama_analysis
from app.file_uploader import FileUploader
from app.db_handler import initialize_db_connection, close_db_connection
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
        ollama_config = config['ollama_config']
        s3_config = config['s3_config']
        db_config = config['db_config']
        visicooler_config = config['visicooler_config']

        # Initialize database connection ONCE
        conn, cur = initialize_db_connection(db_config)

        # Check visibilitydetails schema
        if not check_visibilitydetails_schema(cur):
            logger.error("Schema validation failed for orgi.visibilitydetails.")
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

            # Get cyclecountid for Ollama analysis
            try:
                cur.execute("""
                    SELECT GREATEST(
                        COALESCE((SELECT MAX(cyclecountid) FROM orgi.visibilitydetails), 0),
                        COALESCE((SELECT MAX(stagingid) FROM orgi.visibilityitemsstaging), 0)
                    ) AS max_cycle
                """)
                row = cur.fetchone()
                max_cycle = int(row[0]) if row and row[0] is not None else 0
                cyclecountid = max_cycle + 1
                logger.info(f"Computed cyclecountid: {cyclecountid}")
            except Exception as e:
                logger.error(f"Failed to compute cyclecountid: {e}. Using cyclecountid = 1")
                cyclecountid = 1

            # Run visicooler analysis (handles subcategory 605 - cooler metrics)
            logger.info("Running visicooler analysis for subcategory 605...")
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
                logger.error(f"Visicooler analysis failed: {e}")
                visicooler_records = []

            # Run Ollama analysis (handles other subcategories - visibility items)
            # IMPORTANT: Pass the SAME database connection to avoid connection issues
            logger.info("Running Ollama analysis for visibility items...")
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
                    cyclecountid=cyclecountid,
                    conn=conn,  # PASS EXISTING CONNECTION
                    cur=cur     # PASS EXISTING CURSOR
                )
                logger.info(f"Ollama analysis completed: {len(ollama_results)} records generated.")
            except Exception as e:
                logger.error(f"Ollama analysis failed: {e}")
                logger.error(traceback.format_exc())
                ollama_results, ollama_csv = [], None

            # Upload Ollama CSV to S3
            if ollama_csv and os.path.exists(ollama_csv):
                logger.info("Uploading Ollama CSV to S3...")
                try:
                    s3_handler.upload_file_to_s3(
                        ollama_csv,
                        f"{ollama_config['output_s3_folder']}VisibleItem_{cyclecountid}/analysis_results.csv"
                    )
                    logger.info("Ollama CSV uploaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to upload Ollama CSV to S3: {e}")
            else:
                logger.warning("No Ollama CSV to upload.")

            # Update processed_flag in orgi.fileupload
            logger.info("Updating processed_flag for processed images...")
            try:
                file_uploader = FileUploader(None)
                failed_updates = file_uploader.update_processed_flag(conn, image_paths)
                if failed_updates:
                    logger.warning(f"Failed to update processed_flag for {len(failed_updates)} files")
                else:
                    logger.info(f"Successfully updated processed_flag for {len(image_paths)} files.")
            except Exception as e:
                logger.error(f"Error updating processed_flag: {e}")

            logger.info("=" * 60)
            logger.info("PIPELINE SUMMARY:")
            logger.info(f"Images processed: {len(image_paths)}")
            logger.info(f"Visicooler records: {len(visicooler_records)}")
            logger.info(f"Ollama records: {len(ollama_results) if ollama_results else 0}")
            logger.info(f"Failed files: {len(failed_files)}")
            logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error(traceback.format_exc())
        if conn and 'conn' in locals():
            conn.rollback()
    finally:
        if conn is not None:
            close_db_connection(conn, cur)
        logger.info("Pipeline execution completed.")

if __name__ == "__main__":
    main()