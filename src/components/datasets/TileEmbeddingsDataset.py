from torch.utils.data import Dataset
from src.components.objects.Logger import Logger
import torch


class TileEmbeddingsDataset(Dataset, Logger):
    def __init__(self, df, cohort_to_index=None, transform=None, target_transform=None):
        self.df = df.reset_index(drop=True)
        self.cohort_to_index = cohort_to_index
        self.transform = transform
        self.target_transform = target_transform
        self.log(f"TileEmbeddingsDataset created with {len(self.df)} slides.", log_importance=1)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        tile_embeddings = torch.load(row['path'])
        y = row['y']
        c = self.cohort_to_index[row['cohort']]
        slide_uuid = row['slide_uuid']
        patient_id = row['patient_id']
        path = row['path']
        return tile_embeddings, c, y, slide_uuid, patient_id, path

    def __len__(self):
        return len(self.df)


