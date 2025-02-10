import logging

# Configure logging
logging.basicConfig(
    filename="portfolio.log",  # Log file
    filemode="a",  # Append logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO  # Log INFO and above
)

def get_logger(name):
    """
    Returns a logger instance for the given module.
    Usage:
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
