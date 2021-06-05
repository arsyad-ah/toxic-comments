import os
import configparser
from ast import literal_eval

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return True


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
            except KeyError:
                raise configparser.NoSectionError(section)
        else:
            return super().options(section, **kwargs)