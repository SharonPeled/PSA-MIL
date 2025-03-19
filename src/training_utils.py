import numpy as np
import pandas as pd
from timm.models.vision_transformer import VisionTransformer
import timm
from torch import nn
import torch
from src.components.objects.Logger import Logger
from torchvision import transforms
from src.configs import Configs
from torch.utils.data import DataLoader
from torch.multiprocessing import set_sharing_strategy
import pytorch_lightning as pl
from PIL import Image
from copy import deepcopy
from src.components.models.CohortAwareVisionTransformer import CohortAwareVisionTransformer
from datetime import datetime
from pytorch_lightning.strategies import DDPStrategy
import os
from src.components.objects.SaveAndLogOutputsCallback import SaveAndLogOutputsCallback
from torchvision.models.resnet import Bottleneck, ResNet


def train(df, train_transform, test_transform, logger, callbacks, model, dataset_fn, split_obj=None, **kwargs):
    if Configs.get('PREDEFINED_COHORT_SPLIT') is not None:
        split_obj = create_predefined_cohort_split_obj_generator(df, Configs.get('PREDEFINED_COHORT_SPLIT'))
    elif Configs.get('PREDEFINED_SPLIT_COL') is not None:
        split_obj = create_predefined_split_obj_generator(df, Configs.get('PREDEFINED_SPLIT_COL'))
    elif split_obj is None:
        split_obj = train_test_valid_split_patients_stratified(df,
                                                               stratified_cols=Configs.get('STRATIFIED_COLS'),
                                                               num_folds=Configs.get('NUM_FOLDS'),
                                                               random_seed=Configs.get('RANDOM_SEED'))
    else:
        Logger.log(f'Custom split object is used.', log_importance=1)
    cross_validate(df, split_obj, train_transform, test_transform, logger, model, callbacks, dataset_fn, **kwargs)
    Logger.log(f"Finished.", log_importance=1)


def create_predefined_cohort_split_obj_generator(df, predefined_cohort_split):
    train_inds = df[df.cohort.isin(predefined_cohort_split['TRAIN'])].index.values
    test_inds = df[df.cohort.isin(predefined_cohort_split['TEST'])].index.values
    yield train_inds, test_inds


def create_predefined_split_obj_generator(df, split_col):
    test_splits = df[split_col].unique()
    for test_split in test_splits:
        train_inds = df[~(df[split_col] == test_split)].index.values
        test_inds = df[df[split_col] == test_split].index.values
        yield train_inds, test_inds


def cross_validate(df, split_obj, train_transform, test_transform, logger, model, callbacks, dataset_fn, **kwargs):
    if Configs.get('TASK') == 'classification':
        Logger.log(f"y value counts: {df.y.value_counts().to_dict()}", log_importance=1)

    for i, (train_inds, test_inds) in enumerate(split_obj):

        if Configs.get('CONTINUE_FROM_FOLD') and Configs.get('CONTINUE_FROM_FOLD') > i:
            Logger.log(f"Skipped Fold {i}", log_importance=1)
            continue
        Logger.log(f"Fold {i}", log_importance=1)

        if Configs.get('EXPERIMENT_SAVE_ARTIFACTS_DIR') is not None:
            train_fold_dir = os.path.join(Configs.get('EXPERIMENT_SAVE_ARTIFACTS_DIR'), str(i), 'train')
            test_fold_dir = os.path.join(Configs.get('EXPERIMENT_SAVE_ARTIFACTS_DIR'), str(i), 'test')
            os.makedirs(train_fold_dir, exist_ok=True)
            os.makedirs(test_fold_dir, exist_ok=True)

            if Configs.get('SAVE_MODEL'):
                Configs.set('TRAINED_MODEL_SAVE_PATH', os.path.join(train_fold_dir, 'trained_model_{time}.ckpt'))

            if Configs.get('SAVE_DF_TEST'):
                Configs.set('TEST_DF_OUTPUT_PATH', os.path.join(test_fold_dir, 'df_test_{time}.csv'))

            if Configs.get('SAVE_DF_TRAIN'):
                Configs.set('TRAIN_DF_OUTPUT_PATH', os.path.join(train_fold_dir, 'df_train_{time}.csv'))

            if Configs.get('SAVE_DF_PRED_TEST'):
                Configs.set('TEST_DF_PRED_OUTPUT_PATH', os.path.join(test_fold_dir, 'df_pred_test_{time}.csv'))

            if Configs.get('SAVE_DF_PRED_TRAIN'):
                Configs.set('TRAIN_DF_PRED_OUTPUT_PATH', os.path.join(train_fold_dir, 'df_pred_train_{time}.csv'))

        fold_model = deepcopy(model)
        df_train = df.loc[train_inds].reset_index(drop=True)
        df_test = df.loc[test_inds].reset_index(drop=True)

        time_str = datetime.now().strftime('%d_%m_%Y_%H_%M')
        if Configs.get('TRAIN_DF_OUTPUT_PATH'):
            df_train.to_csv(Configs.get('TRAIN_DF_OUTPUT_PATH').format(time=time_str), index=False)
        if Configs.get('TEST_DF_OUTPUT_PATH'):
            df_test.to_csv(Configs.get('TEST_DF_OUTPUT_PATH').format(time=time_str), index=False)

        _ = train_single_split(df_train, df_test, train_transform, test_transform, logger, fold_model,
                                          callbacks, dataset_fn, **kwargs)

        if Configs.get('SINGLE_FOLD'):
            break

    log_cross_validation_metrics(pd.DataFrame(SaveAndLogOutputsCallback.METRICS),
                                 logger)


def log_cross_validation_metrics(metrics_df, logger):
    metrics_mean = metrics_df.mean().add_suffix('_mean')
    metrics_std = metrics_df.std().add_suffix('_std')
    # Combine mean and std into a single dictionary
    metrics_combined = pd.concat([metrics_mean, metrics_std], axis=0).to_dict()
    # Log each metric (mean and std) using the logger
    for metric_str, metric_val in metrics_combined.items():
        logger.experiment.log_metric(logger.run_id, metric_str, metric_val)
        if metric_str.endswith('_mean'):
            metric = metric_str[:-5]
            mean_val = metrics_combined[metric + '_mean']
            std_val = metrics_combined[metric + '_std']
            print(f"{metric}: {mean_val:.2f} ± {std_val:.2f}")


def train_single_split(df_train, df_test, train_transform, test_transform, logger, model, callbacks, dataset_fn,
                       **kwargs):
    assert not model.is_fit
    Logger.log(f'Single train split started with:', log_importance=1)
    Logger.log(f'Train slides - {df_train.slide_uuid.unique()}', log_importance=0)
    Logger.log(f'Test slides - {df_test.slide_uuid.unique()}', log_importance=0)

    if Configs.get('VALIDATION_SIZE'):
        split_obj = train_test_valid_split_patients_stratified(df_train,
                                                               stratified_cols=Configs.get('STRATIFIED_COLS'),
                                                               num_folds=int(1/Configs.get('VALIDATION_SIZE')),
                                                               random_seed=Configs.get('RANDOM_SEED'))
        train_inds, val_inds = next(split_obj)
        df_val = df_train.loc[val_inds].reset_index(drop=True)
        df_train = df_train.loc[train_inds].reset_index(drop=True)
        _, _, _, val_loader = get_loaders_and_datasets(df_train, df_val, train_transform, test_transform, dataset_fn,
                                                       **kwargs)
        model.init_validation_set(val_loader, Configs.get('VAL_METRIC'), Configs.get('NUM_MODELS'))

    if Configs.get('CONCAT_TRAIN_TO_SINGLE_EPOCH'):
        # to improve running time with many epochs (in MIL)
        Logger.log(f'Concatenating all epochs into a single epoch.', log_importance=1)
        model.epoch_size = np.ceil(len(df_train) / Configs.get('TRAIN_BATCH_SIZE'))
        df_train_list = []
        for i in range(Configs.get('NUM_EPOCHS')):
            df_train_list.append(df_train.sample(frac=1, replace=False))
        df_train = pd.concat(df_train_list, ignore_index=True)

    train_dataset, test_dataset, train_loader, test_loader = get_loaders_and_datasets(df_train, df_test,
                                                                                      train_transform,
                                                                                      test_transform,
                                                                                      dataset_fn,
                                                                                      **kwargs)

    if hasattr(model, 'init_sample_weight'):
        model.init_sample_weight(train_dataset)

    Logger.log("Starting Training.", log_importance=1)
    Logger.log(f"Training loader size: {len(train_loader)}.", log_importance=1)

    save_outputs_callback = SaveAndLogOutputsCallback(Configs.get('TEST_DF_PRED_OUTPUT_PATH'), Configs.get('TASK'),
                                                      logger=logger)
    trainer = pl.Trainer(devices=Configs.get('NUM_DEVICES'),
                         accelerator=Configs.get('DEVICE'),
                         num_nodes=Configs.get('NUM_NODES'),
                         deterministic=False,
                         callbacks=callbacks + [save_outputs_callback, ],
                         enable_checkpointing=False,
                         logger=logger,
                         num_sanity_val_steps=2,
                         max_epochs=1 if Configs.get('CONCAT_TRAIN_TO_SINGLE_EPOCH') else Configs.get('NUM_EPOCHS'),
                         strategy=DDPStrategy(find_unused_parameters=True),
                         reload_dataloaders_every_n_epochs=1,
                         )

    model.init_schedulers(Configs.get('NUM_EPOCHS'), len(train_loader))

    trainer.fit(model, train_loader)
    time_str = datetime.now().strftime('%d_%m_%Y_%H_%M')

    if Configs.get('TRAINED_MODEL_SAVE_PATH'):
        trainer.save_checkpoint(Configs.get('TRAINED_MODEL_SAVE_PATH').format(time=time_str))
        Logger.log('Model saved: {}'.format(Configs.get('TRAINED_MODEL_SAVE_PATH').format(time=time_str)),
                   log_importance=1)

    Logger.log("Done Training.", log_importance=1)
    Logger.log("Starting Test.", log_importance=1)

    trainer.test(model, test_loader)

    Logger.log(f"Done Test.", log_importance=1)

    if Configs.get('TRAIN_DF_PRED_OUTPUT_PATH'):
        Logger.log(f'Inference train', log_importance=1)
        save_outputs_callback.path = Configs.get('TRAIN_DF_PRED_OUTPUT_PATH')
        save_outputs_callback.logger = None
        trainer.test(model, train_loader)

    elif Configs.get('TRAIN_OUTPUT_PATH'):
        train_dataset.df.to_csv(Configs.get('TRAIN_OUTPUT_PATH'), index=False)

    return model


def init_training_transforms(tile_based):
    if not tile_based:
        # MIL
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform, transform

    from src.components.objects.RandStainNA.randstainna import RandStainNA

    train_transform = transforms.Compose([
        transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.75, 1.0), interpolation=Image.BICUBIC), ],
                               p=0.1),

        transforms.Resize(224),

        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),

        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.25, 1)), ], p=0.1),

        # transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2), ], p=0.1),

        transforms.RandomApply([transforms.Grayscale(num_output_channels=3), ], p=0.2),  # grayscale 20% of the images
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),

        transforms.RandomApply([
            RandStainNA(yaml_file=Configs.get('SSL_STATISTICS')['HSV'], std_hyper=0.01, probability=1.0, distribution="normal",
                        is_train=True),
            RandStainNA(yaml_file=Configs.get('SSL_STATISTICS')['HED'], std_hyper=0.01, probability=1.0, distribution="normal",
                        is_train=True),
            RandStainNA(yaml_file=Configs.get('SSL_STATISTICS')['LAB'], std_hyper=0.01, probability=1.0, distribution="normal",
                        is_train=True)],
            p=0.8),

        transforms.Resize(224),
        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform


def lr_scheduler_linspace_steps(lr_pairs, tot_iters):
    left_iters = tot_iters - sum([num_iters for _, num_iters in lr_pairs[:-1] if num_iters > 1])
    lr_array = []
    for (lr, iters), (next_lr, _) in zip(lr_pairs, lr_pairs[1:]):
        if iters == 0:
            continue
        if iters == -1:
            lr_array.append(np.linspace(lr, next_lr, int(left_iters)))
        elif iters < 1:
            lr_array.append(np.linspace(lr, next_lr, int(left_iters*iters)))
            left_iters -= int(left_iters*iters)
        else:
            lr_array.append(np.linspace(lr, next_lr, iters))
    return np.concatenate(lr_array)


def lr_scheduler_cosspace_steps(lr_pairs, tot_iters):
    def cos_space(base_value, final_value, num_iters):
        iters = np.array(range(num_iters))
        # return np.logspace(np.log(start), np.log(stop), num)
        return final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    left_iters = tot_iters - sum([num_iters for _, num_iters in lr_pairs[:-1] if num_iters > 1])
    lr_array = []
    for (lr, iters), (next_lr, _) in zip(lr_pairs, lr_pairs[1:]):
        if iters == 0:
            continue
        if iters == -1:
            lr_array.append(cos_space(lr, next_lr, int(left_iters)))
        elif iters < 1:
            lr_array.append(cos_space(lr, next_lr, int(left_iters*iters)))
            left_iters -= int(left_iters*iters)
        else:
            lr_array.append(cos_space(lr, next_lr, iters))
    return np.concatenate(lr_array)


def set_worker_sharing_strategy(worker_id: int) -> None:
    set_sharing_strategy('file_system')


def get_sample_inverse_proportionate_probs(df):
    df = df.drop_duplicates(subset=['path']).reset_index(drop=True)
    w_per_y_per_cohort = 1 / (df.y.nunique() * df.cohort.nunique())
    slide_counts = df.groupby(['y', 'cohort'], as_index=False).slide_uuid.nunique().rename(
        columns={'slide_uuid': 'num_slides_per_y_per_cohort'})
    slide_counts['slide_w'] = w_per_y_per_cohort / slide_counts.num_slides_per_y_per_cohort
    sample_weight = df.merge(slide_counts[['slide_w', 'y', 'cohort']], on=['y', 'cohort'],
                             how='inner', suffixes=('', '__y'))
    sample_weight['w'] = sample_weight['slide_w'] / sample_weight['slide_w'].sum()
    sample_weight.set_index(sample_weight.path.values, inplace=True)
    Logger.log(f"""AbstractMILClassifier update sampler to WeightedRandomSampler.""", log_importance=1)
    return sample_weight


def get_loaders_and_datasets(df_train, df_test, train_transform, test_transform, dataset_fn, **kwargs):

    train_dataset = dataset_fn(df=df_train, transform=train_transform)
    test_dataset = dataset_fn(df=df_test, transform=test_transform)

    if Configs.get('SLIDE_WEIGHTING_STRATEGY') == 'weighted_sampling':
        from torch.utils.data import WeightedRandomSampler
        sample_weight = get_sample_inverse_proportionate_probs(df_train)
        sampler = WeightedRandomSampler(
            weights=sample_weight.w.loc[df_train.path.values],
            num_samples=len(df_train),
            replacement=True
        )
    else:
        sampler = None

    train_loader = DataLoader(train_dataset, batch_size=Configs.get('TRAIN_BATCH_SIZE'),
                              shuffle=sampler is None,
                              sampler=sampler,
                              persistent_workers=True, num_workers=Configs.get('NUM_WORKERS_TRAIN'),
                              worker_init_fn=set_worker_sharing_strategy,
                              collate_fn=kwargs.get('collate_fn'))

    test_loader = DataLoader(test_dataset, batch_size=Configs.get('TEST_BATCH_SIZE'),
                             shuffle=False,
                             persistent_workers=True, num_workers=Configs.get('NUM_WORKERS_TEST'),
                             worker_init_fn=set_worker_sharing_strategy,
                             collate_fn=kwargs.get('collate_fn'))

    return train_dataset, test_dataset, train_loader, test_loader


def train_test_valid_split_patients_stratified(df, stratified_cols, num_folds, random_seed):
    from sklearn.model_selection import StratifiedGroupKFold
    splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
    if len(stratified_cols) == 0:
        y_to_stratify = np.ones(len(df))
    else:
        y_to_stratify = df[stratified_cols].apply(lambda row: '_'.join(map(str, row)), axis=1).values
    split = splitter.split(X=df, y=y_to_stratify, groups=df['patient_id'])
    return split


def get_ssl_hist_pretrained_url(key):
    url_prefix = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch"
    }
    pretrained_url = f"{url_prefix}/{model_zoo_registry.get(key)}"
    return pretrained_url


def load_ssl_hist_vit_small(pretrained, progress, key):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = get_ssl_hist_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        Logger.log(verbose, log_importance=1)
    return model


def load_ssl_hist_vit_small_cohort_aware(pretrained, progress, key, cohort_aware_dict, **vit_kwargs):
    model = CohortAwareVisionTransformer(
        cohort_aware_dict=cohort_aware_dict,
        img_size=224, patch_size=16, embed_dim=384, num_heads=6, num_classes=0, depth=12, **vit_kwargs
    )
    if pretrained:
        pretrained_url = get_ssl_hist_pretrained_url(key)
        model.load_pretrained_model(torch.hub.load_state_dict_from_url(pretrained_url, progress=progress))
    return model


def load_dino_vit_small_cohort_aware(ckp_path, cohort_aware_dict):
    state_dict = torch.load(ckp_path, map_location="cpu")
    state_dict = state_dict['teacher']
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multi-crop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    model = CohortAwareVisionTransformer(
        cohort_aware_dict=cohort_aware_dict,
        img_size=224, patch_size=16, embed_dim=384, num_heads=6, num_classes=0, depth=12
    )
    model.fc = nn.Identity()
    msg = model.load_state_dict(state_dict, strict=False)
    Logger.log(msg)
    return model


def load_headless_tile_encoder(tile_encoder_name, ckp_path=None, cohort_aware_dict=None):
    if tile_encoder_name == 'IMAGENET_VIT_PRETRAINED':
        model_name = 'vit_small_patch16_224'
        # Load the pre-trained ViT model
        model = timm.create_model(model_name, pretrained=True)
        model.head = nn.Identity()
        cohort_required = False
        return model, model.num_features, cohort_required
    elif tile_encoder_name == 'SSL_VIT_PRETRAINED':
        model = load_ssl_hist_vit_small(pretrained=True, progress=False, key="DINO_p16")
        cohort_required = False
        return model, model.num_features, cohort_required
    elif tile_encoder_name == 'SSL_VIT_PRETRAINED_COHORT_AWARE':  # tile ca?
        model = load_ssl_hist_vit_small_cohort_aware(pretrained=True, progress=False, key="DINO_p16",
                                                     cohort_aware_dict=cohort_aware_dict)
        cohort_required = True
        return model, model.num_features, cohort_required
    elif tile_encoder_name == 'VIT_PRETRAINED_DINO':  # dino CA
        model = load_dino_vit_small_cohort_aware(ckp_path=ckp_path, cohort_aware_dict=cohort_aware_dict)
        cohort_required = True
        return model, model.num_features, cohort_required
    elif tile_encoder_name == 'SSL_RESNET_PRETRAINED':
        model = load_ssl_hist_resnet50(pretrained=True, progress=False, key="BT")
        model.num_features = 1000
        return model, model.num_features, False
    elif tile_encoder_name == 'ImageNet_RESNET_PRETRAINED':
        model, _, _ = load_imagenet_resnet_model()
        model.num_features = 1000
        return model, model.num_features, False


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        return x

def get_resent_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def load_ssl_hist_resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_resent_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model


def load_imagenet_resnet_model():
    model_name = 'resnet50'  # Example: ResNet-50, you can choose other variants like resnet18, resnet34, etc.

    # Load the pre-trained ResNet model
    model = timm.create_model(model_name, pretrained=True)

    # Replace the classification head with an identity layer (if you don't need the classifier)
    model.fc = nn.Identity()  # ResNet uses 'fc' for the final fully connected layer

    cohort_required = False
    model.num_features = 1000

    return model, model.num_features, cohort_required

