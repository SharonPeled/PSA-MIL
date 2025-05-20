from src.components.objects.Logger import Logger
from src.training_utils import lr_scheduler_linspace_steps, lr_scheduler_cosspace_steps
from src.components.models.AbstractClassifier import AbstractClassifier
import numpy as np
import time
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from src.components.models.utils import calc_safe_auc


class AbstractMILClassifier(AbstractClassifier):
    def __init__(self, num_classes, task, learning_rate_pairs, weight_decay_pairs,
                 weighting_strategy=None, log_flops=False):
        super(AbstractMILClassifier, self).__init__(num_classes, task, learning_rate=0,
                                                    weighting_strategy=weighting_strategy)
        self.learning_rate_pairs = learning_rate_pairs
        self.weight_decay_pairs = weight_decay_pairs
        self.lr_list = []
        self.wd_list = []
        self.global_iter = 0
        self.epoch_loss = 0
        self.epoch_size = None
        self.epoch_init_time = None
        self.epoch_times_list = []
        self.val_loader = None
        self.val_metric = None
        self.models_for_bagging = []
        self.num_models_for_bagging = None
        self.log_flops = log_flops
        Logger.log(f"""AbstractMILClassifier created.""", log_importance=1)

    def init_validation_set(self, val_loader, val_metric, num_models_for_bagging):
        self.val_metric = val_metric
        self.val_loader = val_loader
        self.num_models_for_bagging = num_models_for_bagging

    def init_sample_weight(self, train_dataset):
        if self.weighting_strategy != 'slide_balance':
            return super().init_sample_weight(train_dataset)
        df = train_dataset.df.drop_duplicates(subset=['path']).reset_index(drop=True)
        if self.task != 'survival':
            slide_counts = slide_balance_sample_weight(df)
            self.sample_weight = df.merge(slide_counts[['w', 'y', 'cohort']], on=['y', 'cohort'],
                                          how='inner', suffixes=('', '__y'))
        else:
            slide_counts = slide_balance_sample_weight_surv(df)
            self.sample_weight = df.merge(slide_counts[['w', 'cohort']], on=['cohort',],
                                          how='inner', suffixes=('', '__x'))
        self.sample_weight.set_index(self.sample_weight.path.values, inplace=True)
        Logger.log(f"""AbstractMILClassifier update sample weights: {self.weighting_strategy}.""", log_importance=1)

    def on_train_start(self):
        if self.device != 'cpu':
            torch.cuda.reset_peak_memory_stats()
        total_params = sum(p.numel() for p in self.parameters())
        total_params = total_params / 1e6
        self.logger.experiment.log_metric(self.logger.run_id, "tot_params", total_params)
        total_size = total_params * 4  # Assuming 32-bit (4 bytes) floating point numbers
        total_size = round(total_size / (1024 ** 3), 2) # gb
        self.logger.experiment.log_metric(self.logger.run_id, "total_size", total_size)

    def init_schedulers(self, num_epochs, steps_per_epoch):
        tot_steps = num_epochs * steps_per_epoch + 1
        if self.learning_rate_pairs[0] == 0:
            self.lr_list = lr_scheduler_linspace_steps(self.learning_rate_pairs[1], tot_iters=tot_steps)
        elif self.learning_rate_pairs[0] == 1:
            self.lr_list = lr_scheduler_cosspace_steps(self.learning_rate_pairs[1], tot_iters=tot_steps)
        if self.weight_decay_pairs[0] == 0:
            self.wd_list = lr_scheduler_linspace_steps(self.weight_decay_pairs[1], tot_iters=tot_steps)
        elif self.weight_decay_pairs[0] == 1:
            self.wd_list = lr_scheduler_cosspace_steps(self.weight_decay_pairs[1], tot_iters=tot_steps)
        self.init_lr = self.lr_list[0]

    def on_before_optimizer_step(self, optimizer, optimizer_idx=None):
        opt = self.optimizers(use_pl_optimizer=False)

        if self.global_iter == 0:
            self.global_iter += 1
            return

        if not isinstance(opt, list):
            for param_group in opt.param_groups:
                param_group['lr'] *= (self.lr_list[self.global_iter] / self.lr_list[self.global_iter-1])
                param_group['weight_decay'] = self.wd_list[self.global_iter]

        else:
            for opt in self.optimizers(use_pl_optimizer=False):
                for param_group in opt.param_groups:
                    param_group['lr'] *= (self.lr_list[self.global_iter] / self.lr_list[self.global_iter - 1])
                    param_group['weight_decay'] = self.wd_list[self.global_iter]

        self.logger.experiment.log_metric(self.logger.run_id, f"lr", self.lr_list[self.global_iter])
        self.logger.experiment.log_metric(self.logger.run_id, f"wd", self.wd_list[self.global_iter])
        self.global_iter += 1

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # current_memory = torch.cuda.memory_allocated(0)
        # current_memory = round(current_memory / (1024 ** 3), 2)  # gb
        # self.logger.experiment.log_metric(self.logger.run_id, "current_memory", current_memory)
        self.epoch_loss += outputs['loss'].detach().cpu()
        if self.global_iter % self.epoch_size == 0:
            self.manual_on_epoch_end()

    def manual_on_epoch_end(self):
        self.logger.experiment.log_metric(self.logger.run_id, f"epoch_loss",
                                          self.epoch_loss / len(self.trainer.train_dataloader))
        self.epoch_loss = 0

        if self.epoch_init_time is not None:
            self.epoch_times_list.append(time.time() - self.epoch_init_time)
        self.epoch_init_time = time.time()

        if self.val_loader is not None:
            # validation and bagging
            outputs = []
            with torch.no_grad():
                self.eval()
                for batch_idx, batch in enumerate(self.val_loader):
                    batch = self.batch_to_device(batch)
                    outputs.append(self.test_step(batch, batch_idx))
                self.train()
            df_patient = AbstractClassifier.get_summary_df(outputs, self.task)
            mean_val = self.log_validation_after_epoch(df_patient)
            if len(self.models_for_bagging) < self.num_models_for_bagging:
                self.models_for_bagging.append([mean_val, self.global_iter // self.epoch_size,
                                                self.state_dict()])
                Logger.log(f"""Bagging models updated: {[(e[0], e[1]) for e in self.models_for_bagging]}""",
                           log_importance=1)
            else:
                min_index = min(range(len(self.models_for_bagging)),
                                key=lambda i: self.models_for_bagging[i][0])
                if mean_val > self.models_for_bagging[min_index][0]:
                    # Replace the model with the lowest value with the new model
                    self.models_for_bagging[min_index] = [mean_val, self.global_iter // self.epoch_size,
                                                          self.state_dict()]
                Logger.log(f"""Bagging models updated: {[(e[0], e[1]) for e in self.models_for_bagging]}""",
                           log_importance=1)

    def log_validation_after_epoch(self, df_patient):
        if self.val_metric =='total_auc':
            return calc_safe_auc(df_patient.y_true, df_patient.logits)

        df_patient_cohort = df_patient.groupby('cohort').apply(lambda df_group: pd.Series(
            (accuracy_score(df_group.y_true, df_group.y_pred),
             balanced_accuracy_score(df_group.y_true, df_group.y_pred),
             f1_score(df_group.y_true, df_group.y_pred),
             calc_safe_auc(df_group.y_true, df_group.logits)),
            index=['accuracy', 'balanced_accuracy', 'f1', 'auc']
        ))

        suffix = 'epoch'
        logger = self.logger
        res = []

        for cohort, row in df_patient_cohort.iterrows():
            accuracy, balanced_accuracy, f1, auc = row['accuracy'], row['balanced_accuracy'], row['f1'], row['auc']
            res.append(row[self.val_metric])
            logger.experiment.log_metric(logger.run_id, f"patient_C{cohort}_AUC_{suffix}", auc)
            logger.experiment.log_metric(logger.run_id, f"patient_C{cohort}_Accuracy_{suffix}", accuracy)
            logger.experiment.log_metric(logger.run_id, f"patient_C{cohort}_balanced_accuracy_{suffix}",
                                         balanced_accuracy)
            logger.experiment.log_metric(logger.run_id, f"patient_C{cohort}_F1_{suffix}", f1)
        hmean_val = len(res) / np.sum(1.0/np.array(res))
        logger.experiment.log_metric(logger.run_id, f"hmean_val_{suffix}", hmean_val)

        return hmean_val

    def on_train_end(self):
        super(AbstractMILClassifier, self).on_train_end()
        avg_epoch_time = np.mean(self.epoch_times_list)
        self.logger.experiment.log_metric(self.logger.run_id, "avg_epoch_time", avg_epoch_time)
        if self.device != 'cpu':
            peak_memory = torch.cuda.max_memory_allocated()
            peak_memory = round(peak_memory / (1024 ** 3), 2)  # gb
            self.logger.experiment.log_metric(self.logger.run_id, "peak_memory", peak_memory)
            Logger.log(f"""Peak Memory: {peak_memory}""", log_importance=1)

    def test_step(self, batch, batch_idx):
        outputs = super(AbstractMILClassifier, self).test_step(batch, batch_idx)
        if not self.is_training and self.is_fit and len(self.models_for_bagging) > 0:
            outputs_bagging = []
            curr_state_dict = self.state_dict()
            for _, _, model_state_dict in self.models_for_bagging:
                self.load_state_dict(model_state_dict)
                outputs_bagging.append(super(AbstractMILClassifier, self).test_step(batch, batch_idx)['logits'])
            outputs['logits'] = torch.mean(torch.stack(outputs_bagging), dim=0)
            self.load_state_dict(curr_state_dict)
            return outputs
        return outputs

    def batch_to_device(self, batch):
        device = next(self.parameters()).device
        # tile embeddings
        batch[0] = list(batch[0])  # from tuple to list
        for i in range(len(batch[0])):
            batch[0][i] = batch[0][i].to(device)
        batch[1] = batch[1].to(device)  # c
        batch[2] = batch[2].to(device)  # y
        if len(batch) > 6:
            batch[6] = list(batch[6])
            batch[7] = list(batch[7])
            for i in range(len(batch[6])):
                batch[6][i] = batch[6][i].to(device)
                batch[7][i] = batch[7][i].to(device)
        return batch


def slide_balance_sample_weight(df):
    w_per_y_per_cohort = 1 / (df.y.nunique() * df.cohort.nunique())
    slide_counts = df.groupby(['y', 'cohort'], as_index=False).slide_uuid.nunique().rename(
        columns={'slide_uuid': 'num_slides_per_y_per_cohort'})
    slide_counts['slide_w'] = w_per_y_per_cohort / slide_counts.num_slides_per_y_per_cohort

    # clipping weights
    mean_value = np.mean(slide_counts.slide_w)
    std_value = np.std(slide_counts.slide_w)
    dev = 1
    slide_counts['w_clipped'] = np.clip(slide_counts.slide_w, mean_value - dev * std_value,
                                        mean_value + dev * std_value)
    slide_counts['sum_w_per_y_per_cohort'] = slide_counts.w_clipped * slide_counts.num_slides_per_y_per_cohort
    y_sum_w = slide_counts.groupby('y', as_index=False).sum_w_per_y_per_cohort.sum()
    y_sum_w['norm_factor'] = (1 / slide_counts.y.nunique()) / y_sum_w.sum_w_per_y_per_cohort
    slide_counts = slide_counts.merge(y_sum_w, how='inner', on='y')
    slide_counts['w'] = slide_counts['w_clipped'] * slide_counts['norm_factor']
    return slide_counts


def slide_balance_sample_weight_surv(df):
    w_per_cohort = 1 / df.cohort.nunique()
    slide_counts = df.groupby('cohort', as_index=False).slide_uuid.nunique().rename(
        columns={'slide_uuid': 'num_slides_per_cohort'})
    slide_counts['slide_w'] = w_per_cohort / slide_counts.num_slides_per_cohort

    # clipping weights
    mean_value = np.mean(slide_counts.slide_w)
    std_value = np.std(slide_counts.slide_w)
    dev = 1
    slide_counts['w_clipped'] = np.clip(slide_counts.slide_w, mean_value - dev * std_value,
                                        mean_value + dev * std_value)
    slide_counts['w'] = slide_counts['w_clipped']
    return slide_counts


