import json
import logging


class Json_formatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'message': record.getMessage(),
        }
        return json.dumps(log_record)


def initLogger():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger(__name__)
    json_formatter = Json_formatter()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(json_formatter)
    logger.addHandler(stream_handler)
