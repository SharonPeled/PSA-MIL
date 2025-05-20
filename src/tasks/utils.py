import pandas as pd
from src.configs import Configs
from src.components.objects.Logger import Logger
from pytorch_lightning.loggers import MLFlowLogger


def load_df_labels_classification():
    # must have LABEL_COL, patient_id and cohort columns
    df_labels = pd.read_csv(Configs.get('DF_LABELS_PATH'))
    df_labels = df_labels[df_labels[Configs.get('LABEL_COL')].isin(Configs.get('CLASS_TO_IND').keys())]
    df_labels['y'] = df_labels[Configs.get('LABEL_COL')].apply(lambda s: Configs.get('CLASS_TO_IND')[s])
    df_labels['y_to_stratified'] = df_labels[Configs.get('STRATIFIED_COLS')].apply(lambda row:
                                                                                   '_'.join(map(str, row)), axis=1)
    df_labels = df_labels[df_labels.cohort.isin(Configs.get('COHORT_TO_IND').keys())]
    assert df_labels.patient_id.is_unique
    Logger.log(f"df_labels loaded, size: {len(df_labels)}", log_importance=1)
    return df_labels


def _discretize_survival_months(times, censorships, eps=1e-6):
    n_bins = Configs.get('NUM_CLASSES')
    disc_labels, q_bins = pd.qcut(times[censorships < 1], q=n_bins, retbins=True, labels=False)
    q_bins[-1] = times.max() + eps
    q_bins[0] = times.min() - eps
    # assign patients to different bins according to their months' quantiles (on all data)
    # cut will choose bins so that the values of bins are evenly spaced. Each bin may have different frequencies
    disc_labels, q_bins = pd.cut(times, bins=q_bins, retbins=True, labels=False,
                                 right=False, include_lowest=True)
    return disc_labels.values.astype(int)


def load_df_labels_survival():
    df_labels = pd.read_csv(Configs.get('DF_LABELS_PATH'))
    df_labels = df_labels[df_labels.cohort.isin(Configs.get('COHORT_TO_IND').keys())].reset_index(drop=True)
    df_labels[Configs.get('LABEL_SURVIVAL_TIME')] /= 30.0  # transform days to months
    df_labels['y_discrete'] = _discretize_survival_months(df_labels[Configs.get('LABEL_SURVIVAL_TIME')],
                                                          df_labels[Configs.get('LABEL_EVENT_IND')])
    df_labels['y'] = df_labels.apply(lambda row: (row[Configs.get('LABEL_SURVIVAL_TIME')],
                                                  row[Configs.get('LABEL_EVENT_IND')],
                                                  row['y_discrete']
                                                  ), axis=1)
    assert df_labels.patient_id.is_unique
    Logger.log(f"df_labels loaded, size: {len(df_labels)}", log_importance=1)
    return df_labels


def load_df_tile_embeddings_labeled():
    if Configs.get('TASK') == 'classification':
        df_labels = load_df_labels_classification()
    elif Configs.get('TASK') == 'survival':
        df_labels = load_df_labels_survival()
    else:
        raise NotImplementedError
    df_tile_embeddings = pd.read_csv(Configs.get('DF_TILE_EMBEDDINGS_PATH'))
    Logger.log(f"df_tile_embeddings_df loaded, size: {len(df_tile_embeddings)}", log_importance=1)
    df_tile_embeddings_labeled = df_labels.merge(df_tile_embeddings, how='inner', on='patient_id',
                                                 suffixes=('', '_x'))
    df_tile_embeddings_labeled.reset_index(drop=True, inplace=True)
    Logger.log(f"df_tile_embeddings_labeled loaded, size: {len(df_tile_embeddings_labeled)}",
               log_importance=1)
    return df_tile_embeddings_labeled


def init_training_callbacks():
    mlflow_logger = MLFlowLogger(experiment_name=Configs.get('EXPERIMENT_NAME'), run_name=Configs.get('RUN_NAME'),
                                 save_dir=Configs.get('MLFLOW_DIR'),
                                 artifact_location=Configs.get('MLFLOW_DIR'),
                                 log_model='all',
                                 tags={"mlflow.note.content": Configs.get('RUN_DESCRIPTION')})
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, Configs.get('config_filepath'),
                                          artifact_path="configs")
    Logger.log(f"""MLFlow logger initialized, config file logged.""", log_importance=1)
    return mlflow_logger, []



