import logging
from src.datapipeline import DataPipeline
from configparser import ConfigParser
from datetime import datetime

NOW = datetime()
RUN_DATE = NOW.strftime('%Y%m%D')
RUN_TIME = NOW.strftime('%H%M%S')

def main(config):
    
    objective = config.get('DEFAULT', 'objective')

    if objective == 'prepare':
       
        pipeline = DataPipeline(config, RUN_TIME)
        pipeline.read_data()
        pipeline.clean_data()
        pipeline.prepare()


    elif objective == 'train':
        pass
    elif objective == 'inference':
        pass
    else:
        raise ValueError(f'Unknown objective {objective}. Please check!')



if __name__ == '__main__':
    path = './config/config.ini'

    config = ConfigParser()
    config.read(path)

    main(config)