import os
import yaml
import configparser
from ast import literal_eval
from configparser import NoSectionError

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return True

def read_training_params(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            file.close()
            return config
    except Exception as exc:
        raise ValueError(f'Error reading the config file: {exc}') from exc


def create_training_params(config, model_params, model_name, run_time):
    train_params = {
        'model_save_path': config.get('PATHS', 'model_save_path'),
        'run_time': run_time, 
        **model_params['TRAIN'][model_name]
        }
    if model_name == 'BiLSTMClfTF':
        train_params.update({'embedding_path': config.get('PATHS', 'embedding_path')})
    return train_params

def cfg_to_dict(cfg, section_name, no_defaults=False, **kwargs):
    results = {}
    for opt in cfg.options(section_name, no_defaults=no_defaults, **kwargs):
        print(opt, section_name)
        try:
            results[opt] = literal_eval(cfg[section_name].get(opt))
        except:
            results[opt] = cfg[section_name].get(opt)
    return results

class ConfigParser(configparser.ConfigParser):
    """Can get options() without defaults
    """
    def options(self, section, no_defaults=False, **kwargs):
        if no_defaults:
            try:
                return list(self._sections[section].keys())
            except KeyError as exc:
                raise NoSectionError(section) from exc
        else:
            return super().options(section, **kwargs)