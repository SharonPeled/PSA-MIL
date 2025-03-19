import logging
import logging.config
import sys
import datetime


class Logger:
    """
    Wrapper class for python logging.
    In order to filter logs from imported modules we deploy the following log levels mapping:
    WARN (log_importance 0) - All logs, similar to INFO
    ERROR (log_importance 1) - Only high log_importance logs, summary logs and not operative logs
    CRITICAL (log_importance 2) - Only logs that is critical, such as errors or potential errors.
    """
    LOG_IMPORTANCE_MAP = {0: logging.DEBUG,
                          1: logging.INFO,
                          2: logging.WARNING}
    TILE_PROGRESS_LOG_FREQ = 0

    def _log(self, msg, log_importance=1, name=None, **kwargs):
        msg = Logger.add_importance_level_to_msg(msg, log_importance)
        print(msg)
        if name:
            logger = logging.getLogger(name)
            logger.log(msg=msg, level=Logger.LOG_IMPORTANCE_MAP[log_importance], **kwargs)
        else:
            logger = logging.getLogger(type(self).__name__)
            logger.log(msg=msg, level=Logger.LOG_IMPORTANCE_MAP[log_importance], **kwargs)
        Logger.flush_logger(logger)

    @staticmethod
    def log(msg, log_importance=1, **kwargs):
        msg = Logger.add_importance_level_to_msg(msg, log_importance)
        print(msg)
        logging.log(msg=msg, level=Logger.LOG_IMPORTANCE_MAP[log_importance], **kwargs)
        Logger.flush_logger(logging.getLogger())

    @staticmethod
    def set_default_logger(verbose, log_file_args, log_importance, log_format):
        handlers = []

        if verbose == 1:
            handlers.append(logging.FileHandler(*log_file_args))

        if verbose == 2:
            handlers.append(logging.StreamHandler(stream=sys.stdout))

        if verbose == 3:
            handlers.append(logging.FileHandler(*log_file_args))
            handlers.append(logging.StreamHandler(stream=sys.stdout))

        if log_importance == 2:
            handlers.append(logging.StreamHandler(stream=sys.stderr))

        logging.basicConfig(handlers=handlers, level=Logger.LOG_IMPORTANCE_MAP[log_importance], **log_format)

    @staticmethod
    def add_importance_level_to_msg(msg, log_importance):
        now = datetime.datetime.now()
        time_str = now.strftime("%d/%m/%y %H:%M:%S")
        return f"[{time_str}] [{log_importance}] {msg}"

    @staticmethod
    def flush_logger(logger):
        for handle in logger.handlers:
            handle.flush()




