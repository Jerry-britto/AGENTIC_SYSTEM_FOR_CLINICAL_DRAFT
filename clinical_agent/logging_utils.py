import os
import logging

def setup_logger():
    """
    Configures a logger that writes both to the terminal and a file in the logs/ directory.
    """
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    log_file_path = os.path.join("logs", "clinical_agent.log")
    
    # Create or get custom logger
    logger = logging.getLogger("clinical_agent")
    logger.setLevel(logging.INFO)
    
    # Avoid adding duplicate handlers if the logger has already been setup
    if not logger.handlers:
        # Create console handler for terminal output
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        
        # Create file handler for logs/ folder output
        f_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        f_handler.setLevel(logging.INFO)
        
        # Create formatter and add to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
    return logger

# Singleton logger instance
logger = setup_logger()
