import logging
logger = None

def log_message(source, message):
    print('{}:{}'.format(source, message))
    logger.info('{}:{}'.format(source, message))

def init_logger(log_file):
    global logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)