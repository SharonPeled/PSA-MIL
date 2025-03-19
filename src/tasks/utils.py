import pandas as pd
from src.configs import Configs
from src.components.objects.Logger import Logger
from pytorch_lightning.loggers import MLFlowLogger
import os
import numpy as np


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
    # TODO: not entirely the same - I guess patients vs slides
    # cut the data into self.n_bins (4= quantiles)
    n_bins = Configs.get('NUM_CLASSES')
    disc_labels, q_bins = pd.qcut(times[censorships < 1], q=n_bins, retbins=True, labels=False)
    q_bins[-1] = times.max() + eps
    q_bins[0] = times.min() - eps

    # assign patients to different bins according to their months' quantiles (on all data)
    # cut will choose bins so that the values of bins are evenly spaced. Each bin may have different frequncies
    disc_labels, q_bins = pd.cut(times, bins=q_bins, retbins=True, labels=False,
                                 right=False, include_lowest=True)
    return disc_labels.values.astype(int)


def load_df_labels_survival():
    # must have LABEL_COL, patient_id columns
    df_labels = pd.read_csv(Configs.get('DF_LABELS_PATH'))
    # df_labels = df_labels[df_labels.cohort==Configs.get('COHORT')]
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


def load_df_tiles():
    # must have patient_id, slide_uuid, cohort, tile_path columns
    df_tiles = pd.read_csv(Configs.get('DF_TILES_PATH'))

    # filter tiles with too few tile
    if Configs.get('MIN_TILES_PER_SLIDE'):
        tile_counts = df_tiles.groupby('slide_uuid', as_index=False).path.count().rename(columns={
            'path': 'num_tiles_per_slide'
        })
        tile_counts_filter = tile_counts.num_tiles_per_slide < Configs.get('MIN_TILES_PER_SLIDE')
        Logger.log(f'Number of slides filtered due to tile count: {tile_counts_filter.sum()}', log_importance=1)
        tile_counts = tile_counts[~tile_counts_filter]
        df_tiles = df_tiles[df_tiles.slide_uuid.isin(tile_counts.slide_uuid)]

    # filter tiles with too few tumor tiles
    if Configs.get('TUM_FILTER'):
        is_tum_col, tum_score_col, tum_count_threshold, min_tiles_sample_crc, min_tiles_sample_stad, min_tiles_sample_ucec = Configs.get('TUM_FILTER')
        tile_tum_counts = df_tiles.groupby('slide_uuid').apply(lambda df_s: len(df_s[df_s[is_tum_col]])).to_frame(
            name='num_tum_tiles_per_slide').reset_index()
        tile_tum_filter = tile_tum_counts.num_tum_tiles_per_slide < tum_count_threshold
        Logger.log(f'Number of slides filtered due to tum count: {tile_tum_filter.sum()}', log_importance=1)
        tile_tum_counts = tile_tum_counts[~tile_tum_filter]
        df_tiles = df_tiles[df_tiles.slide_uuid.isin(tile_tum_counts.slide_uuid)].reset_index(drop=True)
        # filter tiles
        sample_rate_per_cohort = {'CRC': min_tiles_sample_crc,
                                  'STAD': min_tiles_sample_stad,
                                  'UCEC': min_tiles_sample_ucec}
        top_scores_indices = df_tiles.groupby('slide_uuid').apply(
            lambda df_s: df_s.nlargest(sample_rate_per_cohort[df_s.cohort.iloc[0]], tum_score_col).index.values)
        top_scores_indices = pd.Series(np.concatenate(top_scores_indices.values))
        df_tiles['is_top_score'] = df_tiles.index.isin(top_scores_indices.values)
        df_tiles = df_tiles[((df_tiles[is_tum_col].values)|(df_tiles['is_top_score'].values))]
    df_tiles.reset_index(drop=True, inplace=True)
    # for quick debugging
    # df_tiles = df_tiles.groupby('slide_uuid', as_index=False).apply(lambda d: d.sample(min(len(d), 10)))
    # df_tiles = df_tiles[df_tiles.slide_uuid.isin(df_tiles.slide_uuid.unique()[:10])]
    Logger.log(f"df_tiles loaded, size: {len(df_tiles)}", log_importance=1)
    return df_tiles


def load_df_tile_embeddings_all_folds():
    df_list = []
    artifact_dir = Configs.get('TILE_ARTIFACTS_DIR')
    for fold in range(Configs.get('NUM_FOLDS')):
        if Configs.get('CONTINUE_FROM_FOLD') and Configs.get('CONTINUE_FROM_FOLD') > fold:
            continue
        for dataset_str in ['train', 'test']:
            path = os.path.join(artifact_dir, str(fold), dataset_str, "df_tile_embeddings.csv")
            df = pd.read_csv(path)
            df['fold'] = fold
            df['dataset_str'] = dataset_str
            df_list.append(df)
        if Configs.get('SINGLE_FOLD'):
            break
    df_tile_embeddings_all_folds = pd.concat(df_list, ignore_index=True)
    Logger.log(f"df_tile_embeddings_all_folds loaded, size: {len(df_tile_embeddings_all_folds)}", log_importance=1)
    return df_tile_embeddings_all_folds


def load_labeled_tiles():
    df_labels = load_df_labels_classification()
    df_tiles = load_df_tiles()
    df_labeled_tiles = df_labels.merge(df_tiles, how='inner', on='patient_id', suffixes=('', '_x'))
    Logger.log(f"df_labeled_tiles loaded, size: {len(df_labeled_tiles)}", log_importance=1)
    return df_labeled_tiles


def load_labeled_tile_embeddings_all_folds():
    df_labels = load_df_labels_classification()
    df_tile_embeddings_all_folds = load_df_tile_embeddings_all_folds()
    df_labeled_tile_embeddings_all_folds = df_labels.merge(df_tile_embeddings_all_folds, how='inner', on='patient_id',
                                                           suffixes=('', '_x'))
    Logger.log(f"df_tile_embeddings_all_folds loaded, size: {len(df_labeled_tile_embeddings_all_folds)}",
               log_importance=1)
    return df_labeled_tile_embeddings_all_folds


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
    # for quick debugging
    # df_tile_embeddings_labeled = df_tile_embeddings_labeled.groupby(['cohort', 'y'], as_index=False).apply(lambda d: d.sample(min(len(d), 10)))
    # df_tile_embeddings_labeled = df_tile_embeddings_labeled.groupby(Configs.get('PREDEFINED_SPLIT_COL'), as_index=False).apply(lambda d: d.sample(min(len(d), 10)))
    # df_tile_embeddings_labeled = df_tile_embeddings_labeled.groupby([Configs.get('PREDEFINED_SPLIT_COL'), Configs.get('LABEL_EVENT_IND')], as_index=False).apply(lambda d: d.sample(min(len(d), 40)))
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



