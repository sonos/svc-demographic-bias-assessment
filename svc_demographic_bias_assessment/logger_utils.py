import logging


class Colors:
    HEADER = "\033[01;95m"
    OKBLUE = "\033[01;94m"
    OKCYAN = "\033[01;96m"
    OKGREEN = "\033[01;92m"
    OKYELLOW = "\033[0;33m"
    WARNING = "\033[01;93m"
    FAIL = "\033[01;91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\033[92m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s:%(name)s:%(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
