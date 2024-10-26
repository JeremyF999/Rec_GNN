# main.py

import subprocess
import logging
import sys

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Function to run rec_gnn.py
def run_rec_gnn():
    try:
        logger.info("Starting training with rec_gnn.py...")
        subprocess.run(["python", "rec_gnn.py"], check=True)
        logger.info("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred while running rec_gnn.py: {e}")
        sys.exit(1)

# Function to run rec_inference.py
def run_rec_inference():
    try:
        logger.info("Starting inference with rec_inference.py...")
        subprocess.run(["python", "rec_inference.py"], check=True)
        logger.info("Inference completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred while running rec_inference.py: {e}")
        sys.exit(1)

# Main function to run both scripts
def main():
    run_rec_gnn()
    run_rec_inference()

if __name__ == "__main__":
    main()
