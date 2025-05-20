from src.configs import Configs
from src.tasks.utils import load_df_tile_embeddings_labeled
from src.tasks.utils import init_training_callbacks
from src.training_utils import train as train_general
from src.components.datasets.TileSpatialEmbeddingsDataset import TileSpatialEmbeddingsDataset
from functools import partial
from torchvision import transforms
import torch
from src.components.models.PSAClassifier import PSAClassifier


def train():
    df, split_obj, train_transform, test_transform, logger, callbacks, model, dataset_fn, collate_fn = init_task()
    train_general(df, train_transform, test_transform, logger, callbacks, model, dataset_fn,
                  split_obj=split_obj, collate_fn=collate_fn)


def init_task():
    df_tile_embeddings_labeled = load_df_tile_embeddings_labeled()
    train_transform, test_transform = get_empty_transforms()
    logger, callbacks = init_training_callbacks()
    model = init_model()
    dataset_fn, collate_fn = init_dataset()
    return df_tile_embeddings_labeled, None, train_transform, test_transform, logger, \
           callbacks, model, dataset_fn, collate_fn


def init_dataset():
    dataset_fn = partial(TileSpatialEmbeddingsDataset, cohort_to_index=Configs.get('COHORT_TO_IND'))
    collate_fn = psa_collate
    return dataset_fn, collate_fn


def init_model():
    model = PSAClassifier(num_classes=Configs.get('NUM_CLASSES'),\
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
    return [tile_embeddings, torch.tensor(c), torch.tensor(y), slide_uuid, patient_id, path, row, col, distance, indices]




