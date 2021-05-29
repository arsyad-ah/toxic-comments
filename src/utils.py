import os
from ast import literal_eval

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return True


def _cfg_to_dict(cfg, section_name):
    results = {}
    for opt in cfg.options(section_name):
        try:
            results[opt] = literal_eval(cfg[section_name].get(opt))
        except:
            results[opt] = cfg[section_name].get(opt)

    return results