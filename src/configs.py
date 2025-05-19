import os.path
from dataclasses import dataclass
from src.utils import get_time
import yaml
from src.components.objects.Logger import Logger


@dataclass
class ConfigsSingletonClass:
    def __init__(self):
        self.config_dict = {}

    def deploy_yaml_file(self, config_filepath):
        with open(config_filepath, 'r') as yaml_file:
            self.config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        self.config_dict['config_filepath'] = config_filepath
        self.add_computed_configs()
        Logger.log(self.config_dict, log_importance=1)

    def add_computed_configs(self):
        if self.config_dict.get('EXPERIMENT_SAVE_ARTIFACTS_DIR') and \
                self.config_dict.get('EXPERIMENT_NAME'):
            self.config_dict['EXPERIMENT_SAVE_ARTIFACTS_DIR'] = os.path.join(
                self.config_dict['EXPERIMENT_SAVE_ARTIFACTS_DIR'],
                self.config_dict['EXPERIMENT_NAME'] + '_' + self.config_dict['RUN_NAME']
            )

        if self.config_dict.get('TASK') == 'classification' or self.config_dict.get('TASK') is None:
            # default behaviour - classification
            self.config_dict['NUM_CLASSES'] = len(set(list(Configs.get('CLASS_TO_IND').values())))
            self.config_dict['TASK'] = 'classification'

    def paste_current_time(self):
        time_str = get_time()
        for key, val in self.config_dict.items():
            if isinstance(val, str) and '{time}' in val:
                self.config_dict[key] = val.format(time=time_str)
            elif isinstance(val, list):
                for i, sub_val in enumerate(val):
                    if isinstance(sub_val, str) and '{time}' in sub_val:
                        self.config_dict[key][i] = sub_val.format(time=time_str)

    def get(self, key, default=None):
        return self.config_dict.get(key, default)

    def set(self, key, val):
        self.config_dict[key] = val


Configs = ConfigsSingletonClass()
