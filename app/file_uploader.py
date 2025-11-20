import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileUploader:
    def __init__(self, config=None):
        self.config = config

    def update_processed_flag(self, conn, processed_files):
        """Update processed_flag in orgi.fileupload for processed files."""
        failed_updates = []
        try:
            cur = conn.cursor()
            for file_tuple in processed_files:
                try:
                    # Handle tuples with either 5 or 6 elements
                    if len(file_tuple) == 5:
                        filesequenceid, storename, filename, local_path, s3_key = file_tuple
                    elif len(file_tuple) == 6:
                        filesequenceid, storename, filename, local_path, s3_key, storeid = file_tuple
                    else:
                        logger.error(f"Invalid tuple length for file: {file_tuple}. Expected 5 or 6 elements, got {len(file_tuple)}.")
                        failed_updates.append(file_tuple)
                        continue

                    cur.execute(
                        "UPDATE orgi.fileupload SET processed_flag = 'Y' WHERE filesequenceid = %s",
                        (filesequenceid,)
                    )
                    logger.info(f"Updated processed_flag to 'Y' for filesequenceid {filesequenceid}")
                except Exception as e:
                    logger.error(f"Failed to update processed_flag for filesequenceid {filesequenceid}: {e}")
                    failed_updates.append((filesequenceid, storename, filename))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to update processed_flag: {e}")
            failed_updates.extend([(f[0], f[1], f[2]) for f in processed_files if len(f) in {5, 6}])
        finally:
            if 'cur' in locals():
                cur.close()
        return failed_updates