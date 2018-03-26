import os
import json
dir_path = os.path.dirname(os.path.realpath(__file__))
def load_config():
    load_config.config = getattr(load_config, 'config', None)

    if load_config.config is None:
        with open(dir_path + '/config.json') as f:
            config = ''.join(f.readlines())
            f.close()
        load_config.config = json.loads(config)

    return load_config.config


