import pandas as pd

from src.tasks.save_experiment_specific_tile_embeddings import *
from src.tasks.utils import load_df_tiles


def save_tile_embeddings():
    test_transform = get_test_transform()
    save_dir = Configs.get('EXPERIMENT_SAVE_ARTIFACTS_DIR')
    slide_path_list = []
    df_tiles = load_df_tiles()
    df_tiles.set_index(df_tiles.slide_uuid.values, inplace=True)
    df_tiles.sort_values(by=['slide_uuid', 'path'], inplace=True)
    tile_counts = df_tiles.groupby('slide_uuid').path.count().sort_index()
    backbone, cohort_required = init_backbone(None)
    with torch.no_grad():
        backbone = backbone.to(Configs.get('DEVICE_ID'))
        dataset = TileDataset(df_tiles, transform=test_transform,
                              cohort_to_index=Configs.get('COHORT_TO_IND'),
                              obj_to_load=('img', 'cohort', 'slide_uuid', 'path'))
        loader = DataLoader(dataset, batch_size=Configs.get('BATCH_SIZE'),
                            shuffle=False,
                            num_workers=Configs.get('NUM_WORKERS'))

        slide_ind, slide_uuid, tot_num_tiles, df_slide, tile_embed_list, num_tiles_done, path_batch_list = init_next_slide(
            slide_ind=-1,
            tile_counts=tile_counts,
            df_pred=df_tiles)
        for x_all, c_all, batch_slide_uuid_all, batch_path_all in tqdm(loader, total=len(loader)):
            x_all = x_all.to(Configs.get('DEVICE_ID'))
            c_all = c_all.to(Configs.get('DEVICE_ID'))
            batch_slide_uuid_all = np.array(batch_slide_uuid_all)
            batch_path_all = np.array(batch_path_all)
            if cohort_required:
                x_embed_all = backbone(x_all, c_all).detach().cpu()
            else:
                x_embed_all = backbone(x_all).detach().cpu()
            batch_slide_uuid_set = sorted(set(batch_slide_uuid_all))
            print()
            print(pd.Series(batch_slide_uuid_all).value_counts())
            for slide_uuid_sub_iter in batch_slide_uuid_set:
                slide_uuid_filter = batch_slide_uuid_all == slide_uuid_sub_iter
                x_embed = x_embed_all[slide_uuid_filter]
                c = c_all[slide_uuid_filter]
                batch_slide_uuid = batch_slide_uuid_all[slide_uuid_filter]
                batch_path = batch_path_all[slide_uuid_filter]
                print(slide_uuid, len(batch_slide_uuid))

                # rest of computation

                num_tiles_done, x_embed_curr, path_batch_list = add_current_batch(x_embed, tot_num_tiles, num_tiles_done,
                                                                 tile_embed_list,
                                                                 batch_slide_uuid, slide_uuid,
                                                                 batch_path, path_batch_list)

                if num_tiles_done == tot_num_tiles:
                    save_slide_tile_embeddings(tile_embed_list, df_slide, save_dir, slide_path_list, path_batch_list)
                    if len(tile_counts) == slide_ind + 1:
                        continue

                    slide_ind, slide_uuid, tot_num_tiles, df_slide, tile_embed_list, num_tiles_done, path_batch_list = init_next_slide(
                        slide_ind=slide_ind,
                        tile_counts=tile_counts,
                        df_pred=df_tiles)

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
                                save_slide_tile_embeddings(tile_embed_list, df_slide, save_dir, slide_path_list, path_batch_list)
                                if len(tile_counts) == slide_ind + 1:
                                    break

                                slide_ind, slide_uuid, tot_num_tiles, df_slide, tile_embed_list, num_tiles_done, path_batch_list = init_next_slide(
                                    slide_ind=slide_ind,
                                    tile_counts=tile_counts,
                                    df_pred=df_tiles)

    df_slide_paths = pd.DataFrame(slide_path_list, columns=['patient_id', 'slide_uuid', 'path'])
    df_slide_paths.to_csv(os.path.join(save_dir, 'df_tile_embeddings.csv'), index=False)

