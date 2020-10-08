import logging
import logging.handlers
import os

def init_logging(level=logging.DEBUG):
    FORMAT = "[%(asctime)-15s] %(levelname)s  %(message)s"
    logging.basicConfig(format=FORMAT, level=level)

    lodsDir = 'logs'
    if not os.path.exists(lodsDir):
        os.makedirs(lodsDir)

    LOG_FILENAME = os.path.join(lodsDir, 'broccole.log')
    handler = logging.handlers.RotatingFileHandler(
                LOG_FILENAME, maxBytes=10*1024*1024, backupCount=10)
    formatter = logging.Formatter(FORMAT)
    handler.setFormatter(formatter)

    # root logger
    logger = logging.getLogger(None)
    logger.addHandler(handler)
