import logging

level = logging.DEBUG
log_format = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"


class LogManage(object):

    def __init__(self):
        self.logging = logging
        self.logging.basicConfig(format=log_format, level=level)
        self.logger = None

    def get_logger(self, filename, log_name):
        self.logger = self.logging.getLogger(log_name)
        self.logger.setLevel(level)
        self.handler = self.logging.FileHandler(filename)
        self.handler.setLevel(level)
        self.handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(self.handler)
        return self.logger

    def get_logger_print(self, log_name):
        self.logger = self.logging.getLogger(log_name)
        self.logger.setLevel(level)
        return self.logger
