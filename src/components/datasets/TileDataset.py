from torch.utils.data import Dataset
from src.components.objects.Logger import Logger
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class TileDataset(Dataset, Logger):
    def __init__(self, df, obj_to_load=('img', ), cohort_to_index=None, transform=None, target_transform=None,
                 num_mini_epochs=0):
        assert 'img' in obj_to_load and 'path' in df.columns
        self.df = df.reset_index(drop=True)
        self.obj_to_load = obj_to_load
        self.cohort_to_index = cohort_to_index
        self.transform = transform
        self.target_transform = target_transform
        if 'y' in obj_to_load:
            self.df.y = self.df.y.astype(float)
        self.dataset_full_length = len(self.df.index)
        self.dataset_length = len(self.df.index)
        self.num_mini_epochs = num_mini_epochs
        self.index_shift = 0
        self.init_mini_epochs()
        self.log(f"TileDataset created with {self.df.slide_uuid.nunique()} slides, and {len(self.df)} tiles.",
                 log_importance=1)

    def init_mini_epochs(self):
        if self.num_mini_epochs < 2:
            return
        self.dataset_length = self.dataset_full_length // self.num_mini_epochs
        self.index_shift = 0

    def set_mini_epoch(self, epoch, shuffle=False, random_seed=None):
        if self.num_mini_epochs < 2:
            return
        self.index_shift = self.dataset_length * (epoch % self.num_mini_epochs)
        if self.index_shift == 0 and shuffle:
            # so that each epoch will have different shuffle
            self.df = self.df.sample(frac=1, random_state=random_seed + epoch)
        Logger.log(f"Mini epoch number {epoch}, index_shift: {self.index_shift}.")

    def __getitem__(self, index):
        index += self.index_shift
        row = self.df.iloc[index]
        obj_dict = self.load_objs(row)
        return [obj_dict[key] for key in self.obj_to_load]

    def load_objs(self, row):
        obj_dict = {}
        img = self.load_image_safe(row['path'])
        if self.transform:
            img = self.transform(img)
        obj_dict['img'] = img
        if self.cohort_to_index:
            obj_dict['cohort'] = self.cohort_to_index[row['cohort']]
        obj_dict['path'] = row['path']
        if set(self.obj_to_load).issubset(set(obj_dict.keys())):
            # useful in pretraining
            return obj_dict
        if 'y' in self.obj_to_load:
            y = row['y']
            if self.target_transform:
                y = self.target_transform(y)
            obj_dict['y'] = y
        obj_dict['slide_uuid'] = row['slide_uuid']
        obj_dict['patient_id'] = row['patient_id']
        return obj_dict

    def load_image_safe(self, path):
        try:
            # Try to open the image from the given path
            image = Image.open(path)
        except Exception as e:
            # If loading the image fails, create an empty (white) image
            image = Image.new('RGB', (224, 224), color='white')
            self.log('-'*25 + f"Invalid image: {path}, Error: {e}"+ '-'*25,
                     log_importance=1)
        return image

    def __len__(self):
        return self.dataset_length
