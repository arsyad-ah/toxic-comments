import logging
from src.datapipeline import DataPipeline
from src.train import train
from src.inference import make_inference
from configparser import ConfigParser
from datetime import datetime


NOW = datetime.now()
RUN_DATE = NOW.strftime('%Y%m%D')
RUN_TIME = NOW.strftime('%H%M%S')

def main(config):
    
    objective = config.get('DEFAULT', 'objective')
    print('objective: ' , objective)

    if objective == 'prepare':
       
        pipeline = DataPipeline(config, RUN_TIME)
        print('Reading data')
        pipeline.read_data('raw_data')
        print('Cleaning data')
        pipeline.clean_data()
        print('Preparing data')
        pipeline.prepare()

    elif objective == 'train':
        train(config, RUN_TIME)
    elif objective == 'inference':
        make_inference(config, RUN_TIME)
    else:
        raise ValueError(f'Unknown objective {objective}. Please check!')



if __name__ == '__main__':
    import argparse
    from configparser import ConfigParser, ExtendedInterpolation

    parser = argparse.ArgumentParser(
        prog='toxic',
        description='Params to run main function')

    parser.add_argument(
        "--config_path",
        help="Full path to config file",
        default='./config/config.ini',
    )

    args = parser.parse_args()
    config = ConfigParser(interpolation=ExtendedInterpolation())
    print('config load ok')
    config.read(args.config_path)

    print('executing run')
    main(config)