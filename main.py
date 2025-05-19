import argparse
from src.utils import set_global_configs
from src.components.objects.Logger import Logger
from src.configs import Configs
import matplotlib
import warnings
from src.tasks.train_MIL_classifier import train
warnings.filterwarnings('ignore')
matplotlib.use('agg')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_filepath', type=str, required=True)

    args = parser.parse_args()

    Configs.deploy_yaml_file(args.config_filepath)

    set_global_configs(verbose=Configs.get('VERBOSE'),
                       log_file_args=Configs.get('PROGRAM_LOG_FILE_ARGS'),
                       log_importance=Configs.get('LOG_IMPORTANCE'),
                       log_format=Configs.get('LOG_FORMAT'),
                       random_seed=Configs.get('RANDOM_SEED'))

    Logger.log(f"Starting experiment: {Configs.config_dict['EXPERIMENT_NAME']}.{Configs.config_dict['RUN_NAME']}",
               log_importance=1)
    Logger.log(f'Complete config file:\n{Configs.config_dict}')


    train()

    Logger.log(f'Finished!')

if __name__ == "__main__":
    main()


