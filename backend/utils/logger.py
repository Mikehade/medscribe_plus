import logging
import colorlog


def get_logger():
    logger = logging.getLogger(__name__)

    # Check if the logger is already configured
    if not logger.handlers:
        logger.setLevel(
            logging.DEBUG
        )  # Set the desired log level for the entire project

        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s%(reset)s:%(asctime)s - %(message)s",  # Add %(reset)s to reset color after levelname
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "reset",
                "INFO": "cyan",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        # Create a handler for console output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(
            logging.DEBUG
        )  # Set the desired log level for console output
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

    return logger
