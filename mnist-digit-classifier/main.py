from src import rest
from src.utils import logger
from src.models import network

if __name__ == '__main__':
    logger.initLogger()
    model = network.Network(model_type="pytorch")
    rest.run(model)

