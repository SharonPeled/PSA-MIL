import numpy as np
from src.configs import Configs
import pandas as pd
import torch
from src.components.objects.Logger import Logger
from torchvision import transforms
from torch.utils.data import DataLoader
from src.components.datasets.TileDataset import TileDataset
from tqdm import tqdm
import os
from src.training_utils import load_headless_tile_encoder


def save_experiment_specific_tile_embeddings():
    test_transform = get_test_transform()
    experiment_dir = Configs.get('EXPERIMENT_SAVE_ARTIFACTS_DIR')
    for fold in os.listdir(experiment_dir):
        train_dir = os.path.join(experiment_dir, fold, 'train')
        test_dir = os.path.join(experiment_dir, fold, 'test')
        model_path = [os.path.join(train_dir, file) for file in os.listdir(train_dir) if file.endswith(".ckpt")]
        train_pred_path = [os.path.join(train_dir, file) for file in os.listdir(train_dir) if
                           file.startswith("df_train")]
        test_pred_path = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.startswith("df_test")]
        assert len(model_path) <= 1 and len(train_pred_path) == 1 and len(test_pred_path) == 1
        train_pred_path = train_pred_path[0]
        test_pred_path = test_pred_path[0]
        if len(model_path) == 1:
            backbone, cohort_required = init_backbone(model_path[0])
        else:
            backbone, cohort_required = init_backbone(None)
            Logger.log(f"No model found in saving job! Loading the pretrained version!", log_importance=1)
        for pred_path in [train_pred_path, test_pred_path]:

            slide_path_list = []
            df_pred = pd.read_csv(pred_path)
            df_pred.set_index(df_pred.slide_uuid.values, inplace=True)
            df_pred.sort_values(by=['slide_uuid', 'path'], inplace=True)
            tile_counts = df_pred.groupby('slide_uuid').path.count().sort_index()

            with torch.no_grad():
                backbone = backbone.to(Configs.get('DEVICE_ID'))
                dataset = TileDataset(df_pred, transform=test_transform,
                                      cohort_to_index=Configs.get('COHORT_TO_IND'),
                                      obj_to_load=('img', 'cohort', 'slide_uuid', 'path'))
                loader = DataLoader(dataset, batch_size=Configs.get('BATCH_SIZE'),
                                    shuffle=False,
                                    num_workers=Configs.get('NUM_WORKERS'))

                slide_ind, slide_uuid, tot_num_tiles, df_slide, tile_embed_list, num_tiles_done, path_batch_list = init_next_slide(
                    slide_ind=-1,
                    tile_counts=tile_counts,
                    df_pred=df_pred)
                for x, c, batch_slide_uuid, batch_path in tqdm(loader, total=len(loader)):
                    x = x.to(Configs.get('DEVICE_ID'))
                    c = c.to(Configs.get('DEVICE_ID'))
                    batch_slide_uuid = np.array(batch_slide_uuid)
                    if cohort_required:
                        x_embed = backbone(x, c).detach().cpu()
                    else:
                        x_embed = backbone(x).detach().cpu()
                    num_tiles_done, x_embed_curr, path_batch_list = add_current_batch(x_embed, tot_num_tiles, num_tiles_done,
                                                                     tile_embed_list,
                                                                     batch_slide_uuid, slide_uuid,
                                                                     batch_path, path_batch_list)

                    if num_tiles_done == tot_num_tiles:
                        save_slide_tile_embeddings(tile_embed_list, df_slide, os.path.dirname(pred_path),
                                                   slide_path_list, path_batch_list)
                        if len(tile_counts) == slide_ind + 1:
                            continue

                        slide_ind, slide_uuid, tot_num_tiles, df_slide, tile_embed_list, num_tiles_done, path_batch_list = init_next_slide(
                            slide_ind=slide_ind,
                            tile_counts=tile_counts,
                            df_pred=df_pred)

                        if x_embed_curr.shape[0] == x_embed.shape[0]:
                            continue
                        else:
                            rest_slide_uuids = sorted(set(batch_slide_uuid[x_embed_curr.shape[0]:]))
                            assert len(rest_slide_uuids) == 1  # currently only works with single cross batch
                            for slide_uuid in rest_slide_uuids:
                                batch_slide_uuid = batch_slide_uuid[x_embed_curr.shape[0]:]
                                batch_path = batch_path[x_embed_curr.shape[0]:]
                                x_embed = x_embed[x_embed_curr.shape[0]:]
                                num_tiles_done, x_embed_curr, path_batch_list = add_current_batch(x_embed, tot_num_tiles, num_tiles_done,
                                                                                 tile_embed_list,
                                                                                 batch_slide_uuid, slide_uuid,
                                                                                 batch_path, path_batch_list)
                                if num_tiles_done == tot_num_tiles:
                                    save_slide_tile_embeddings(tile_embed_list, df_slide, os.path.dirname(pred_path),
                                                               slide_path_list, path_batch_list)
                                    if len(tile_counts) == slide_ind + 1:
                                        break

                                    slide_ind, slide_uuid, tot_num_tiles, df_slide, tile_embed_list, num_tiles_done, path_batch_list = init_next_slide(
                                        slide_ind=slide_ind,
                                        tile_counts=tile_counts,
                                        df_pred=df_pred)

            df_slide_paths = pd.DataFrame(slide_path_list, columns=['patient_id', 'slide_uuid', 'path'])
            df_slide_paths.to_csv(os.path.join(os.path.dirname(pred_path), 'df_tile_embeddings.csv'), index=False)


def init_next_slide(slide_ind, tile_counts, df_pred):
    slide_ind += 1
    slide_uuid = tile_counts.index[slide_ind]
    tot_num_tiles = tile_counts.iloc[slide_ind]
    df_slide = df_pred.loc[slide_uuid]
    df_slide.sort_values('path', inplace=True)
    tile_embed_list = []
    num_tiles_done = 0
    path_batch_list = []
    return slide_ind, slide_uuid, tot_num_tiles, df_slide, tile_embed_list, num_tiles_done, path_batch_list


def add_current_batch(x_embed, tot_num_tiles, num_tiles_done, tile_embed_list, batch_slide_uuid, slide_uuid,
                      batch_path, path_batch_list):
    x_embed_curr = x_embed[: tot_num_tiles - num_tiles_done]
    tile_embed_list.append(x_embed_curr)

    # batch_path, path_batch_list
    path_curr = batch_path[: tot_num_tiles - num_tiles_done]
    path_batch_list.append(path_curr)

    num_tiles_done += x_embed_curr.shape[0]
    assert (batch_slide_uuid[: x_embed_curr.shape[0]] == slide_uuid).all()

    return num_tiles_done, x_embed_curr, path_batch_list


def save_slide_tile_embeddings(tile_embed_list, df_slide, pred_path, slide_path_list, path_batch_list):
    slide_uuid = df_slide.slide_uuid.iloc[0]
    patient_id = df_slide.patient_id.iloc[0]

    slide_tile_embed = torch.cat(tile_embed_list)
    path = os.path.join(pred_path, 'tile_embeddings', slide_uuid,
                        'tile_embeddings.tensor')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(slide_tile_embed, path)

    slide_tile_paths = np.concatenate(path_batch_list)
    assert (slide_tile_paths == df_slide.path.values).all()  # that the embeddings in same order as df
    df_slide.to_csv(os.path.join(os.path.dirname(path), 'df_slide.csv'), index=False)
    slide_path_list.append((patient_id, slide_uuid, path))


def get_test_transform():
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return test_transform


def init_backbone(path):
    if path:
        if Configs.get('TILE_ENCODER_LOADING_ARGS')['tile_encoder_name'] == 'VIT_PRETRAINED_DINO':
            from src.components.models.CohortAwareTileClassifier import CohortAwareTileClassifier
            model = CohortAwareTileClassifier.load_from_checkpoint(
                path,
                tile_encoder_loading_args=Configs.get('TILE_ENCODER_LOADING_ARGS'),
                num_classes=1,
                learning_rate=None,
                num_iters_warmup_wo_backbone=None,
                weighting_strategy=None)

        else:
            from src.components.models.TileClassifier import TileClassifier
            model = TileClassifier.load_from_checkpoint(path,
                                                        tile_encoder_loading_args=Configs.get('TILE_ENCODER_LOADING_ARGS'),
                                                        num_classes=1,
                                                        learning_rate=None,
                                                        num_iters_warmup_wo_backbone=None,
                                                        weighting_strategy=None)
        Logger.log(f"Backbone successfully loaded from checkpoint!", log_importance=1)
        return model.backbone, model.cohort_required
    else:
        backbone, num_features, cohort_required = load_headless_tile_encoder(**Configs.get('TILE_ENCODER_LOADING_ARGS'))
        return backbone, cohort_required

