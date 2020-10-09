import logging
import logging.handlers
import os

def init_logging(filename='broccole.log', level=logging.DEBUG):
    FORMAT = "[%(asctime)-15s] %(levelname)s  %(message)s"
    logging.basicConfig(format=FORMAT, level=level)

    logsDir = 'logs'
    if not os.path.exists(logsDir):
        os.makedirs(logsDir)

    LOG_FILENAME = os.path.join(logsDir, filename)
    handler = logging.handlers.RotatingFileHandler(
                LOG_FILENAME, maxBytes=10*1024*1024, backupCount=10)
    formatter = logging.Formatter(FORMAT)
    handler.setFormatter(formatter)

    # root logger
    logger = logging.getLogger(None)
    logger.addHandler(handler)
