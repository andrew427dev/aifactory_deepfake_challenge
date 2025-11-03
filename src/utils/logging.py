import logging, sys
def get_logger(name: str="app", level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers: return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)
    return logger
