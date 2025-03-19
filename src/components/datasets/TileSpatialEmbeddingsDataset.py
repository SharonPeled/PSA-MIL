import os.path
import pandas as pd
from src.components.datasets.TileEmbeddingsDataset import TileEmbeddingsDataset
import torch


class TileSpatialEmbeddingsDataset(TileEmbeddingsDataset):
    """
    Assumes the coordinates is in the tile path
    """
    def __init__(self, df, cohort_to_index=None, transform=None, target_transform=None):
        super(TileSpatialEmbeddingsDataset, self).__init__(df=df, cohort_to_index=cohort_to_index,
                                                           transform=transform,
                                                           target_transform=target_transform)
        self.df['slide_df_path'] = self.df.path.apply(lambda p: os.path.join(os.path.dirname(p), 'df_slide.csv'))
        self.df_slides_dict = {row['slide_uuid']: pd.read_csv(row['slide_df_path']) for _, row in
                               self.df.drop_duplicates(subset=['slide_uuid']).iterrows()}
        for slide_uuid, slide_df in self.df_slides_dict.items():
            slide_df['row'] = slide_df.path.apply(lambda s: int(os.path.basename(s).split('_')[0]))
            slide_df['col'] = slide_df.path.apply(lambda s: int(os.path.basename(s).split('_')[1]))
        self.log(f"TileSpatialEmbeddingsDataset created with {len(self.df)} slides.", log_importance=1)

    def __getitem__(self, index):
        tile_embeddings, c, y, slide_uuid, patient_id, path = super().__getitem__(index)
        df_slide = self.df_slides_dict[slide_uuid]
        row = torch.from_numpy(df_slide['row'].values)
        col = torch.from_numpy(df_slide['col'].values)
        points = torch.stack((row.float(), col.float()), dim=1)
        distance = torch.cdist(points, points, p=2)
        sorted_indices = torch.argsort(distance, dim=1)  # Shape: (N, N)
        return tile_embeddings, c, y, slide_uuid, patient_id, path, row, col, distance.numpy(), sorted_indices.numpy()

    def __len__(self):
        return len(self.df)


