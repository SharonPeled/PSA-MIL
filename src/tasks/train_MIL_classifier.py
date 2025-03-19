from src.configs import Configs
from src.tasks.utils import load_df_tile_embeddings_labeled
from src.tasks.utils import init_training_callbacks
from src.training_utils import train as train_general
from src.components.datasets.TileEmbeddingsDataset import TileEmbeddingsDataset
from src.components.datasets.TileSpatialEmbeddingsDataset import TileSpatialEmbeddingsDataset
from functools import partial
from torchvision import transforms
import torch


def train(model_name):
    df, split_obj, train_transform, test_transform, logger, callbacks, model, dataset_fn, collate_fn = init_task(model_name)
    train_general(df, train_transform, test_transform, logger, callbacks, model, dataset_fn,
                  split_obj=split_obj, collate_fn=collate_fn)


def init_task(model_name):
    df_tile_embeddings_labeled = load_df_tile_embeddings_labeled()
    train_transform, test_transform = get_empty_transforms()
    logger, callbacks = init_training_callbacks()
    model = init_model(model_name)
    dataset_fn, collate_fn = init_dataset(model_name)
    # split_obj = create_split_obj_generator(df_labeled_tile_embeddings_all_folds, Configs.get('NUM_FOLDS'))
    return df_tile_embeddings_labeled, None, train_transform, test_transform, logger, \
           callbacks, model, dataset_fn, collate_fn


def init_dataset(model_name):
    dataset_fn = partial(TileSpatialEmbeddingsDataset, cohort_to_index=Configs.get('COHORT_TO_IND'))
    collate_fn = psa_collate
    return dataset_fn, collate_fn


def init_model(model_name):
    if model_name.upper() == 'PSA':
        from src.components.models.PSAClassifier import PSAClassifier
        model = PSAClassifier(num_classes=Configs.get('NUM_CLASSES'),
                              task=Configs.get('TASK'),
                              embed_dim=Configs.get('EMBED_DIM'),
                              attn_dim=Configs.get('ATTN_DIM'),
                              num_heads=Configs.get('NUM_HEADS'),
                              depth=Configs.get('DEPTH'),
                              num_layers_adapter=Configs.get('NUM_RESIDUAL_LAYERS'),
                              patch_drop_rate=Configs.get('PATCH_DROP_RATE'),
                              qkv_bias=Configs.get('QKV_BIAS'),
                              learning_rate_pairs=eval(str(Configs.get('LEARNING_RATE_PAIRS')).
                                                       replace("'", "").replace('null', 'None')),
                              weight_decay_pairs=eval(str(Configs.get('WEIGHT_DECAY_PAIRS')).
                                                      replace("'", "").replace('null', 'None')),
                              weighting_strategy=Configs.get('SLIDE_WEIGHTING_STRATEGY'),
                              pool_type=Configs.get('POOL_TYPE'),
                              reg_terms=Configs.get('REG_TERMS'))
        return model
    raise NotImplementedError


def get_empty_transforms():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform, transform


def create_split_obj_generator(df_all_folds, num_folds):
    for fold in range(num_folds):
        train_inds = df_all_folds[(df_all_folds['fold'] == fold) &
                                  (df_all_folds['dataset_str'] == 'train')].index.values
        test_inds = df_all_folds[(df_all_folds['fold'] == fold) &
                                 (df_all_folds['dataset_str'] == 'test')].index.values
        yield train_inds, test_inds


def collate(batch):
    tile_embeddings, c, y, slide_uuid, patient_id, path = zip(*batch)
    return [tile_embeddings, torch.tensor(c), torch.tensor(y), slide_uuid, patient_id, path]


def psa_collate(batch):
    tile_embeddings, c, y, slide_uuid, patient_id, path, row, col, distance, indices = zip(*batch)
    # print(distance[0].device, indices[0].device)
    return [tile_embeddings, torch.tensor(c), torch.tensor(y), slide_uuid, patient_id, path, row, col, distance, indices]




