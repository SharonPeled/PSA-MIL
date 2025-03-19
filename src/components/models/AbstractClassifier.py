import torch
import pytorch_lightning as pl
from src.components.objects.Logger import Logger
import numpy as np
import pandas as pd
from src.components.models.utils import calc_safe_auc, _calculate_risk
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import torch.nn.functional as F
from src.components.objects.NLLSurvLoss import NLLSurvLoss
from sklearn.metrics import f1_score, precision_recall_curve


class AbstractClassifier(pl.LightningModule):
    """
    Logic that fits both tile based framework and MIL framework.
    """
    def __init__(self, num_classes, task, learning_rate, weighting_strategy):
        super().__init__()
        self.num_classes = 1 if num_classes is not None and num_classes == 2 else num_classes
        self.task = task if task is not None else 'classification'
        self.learning_rate = learning_rate
        self.weighting_strategy = weighting_strategy
        self.sample_weight = None
        self.is_fit = False
        self.is_training = False
        self.is_sample_weight_init = False
        Logger.log(f"""AbstractClassifier created.""", log_importance=1)
    
    def init_sample_weight(self, train_dataset):
        df = train_dataset.df.drop_duplicates(subset=['path']).reset_index(drop=True)

        if self.weighting_strategy == 'class_balance':
            cls_freq = df.y.value_counts().to_dict()
            self.sample_weight = df[['path', 'y']]
            self.sample_weight['w'] = self.sample_weight.y.apply(lambda y: 1.0 / cls_freq[y])
        else:
            self.sample_weight = df[['path']]
            self.sample_weight['w'] = 1

        self.sample_weight.set_index(self.sample_weight.path.values, inplace=True)
        self.is_sample_weight_init = True
        Logger.log(f"""AbstractClassifier update sample weights: {self.weighting_strategy}.""", log_importance=1)
        
    def on_train_start(self):
        assert self.is_sample_weight_init
        self.is_training = True

    def on_train_end(self):
        self.is_fit = True
        self.is_training = False

    def _forward(self, batch):
        raise NotImplementedError

    def forward(self, batch):
        """
        :param batch is a list of params (x, c, path ...)
        In this way each class could require its own forward parameters
        :return:
        """
        scores = self._forward(batch)
        if self.task == 'classification':
            return self._get_logits_from_scores(scores)
        elif self.task == 'survival':
            return scores

    def _get_logits_from_scores(self, scores):
        if len(scores.shape) == 2 and scores.shape[1] > 1:
            logits = F.softmax(scores, dim=-1)
        else:
            logits = F.sigmoid(scores)
        return logits

    def loss_classification(self, logits, y, path):
        w = torch.Tensor(self.sample_weight.loc(axis=0)[list(path)].w.values).to(logits.device).to(
            logits.dtype)
        w = w / w.sum()  # batch weights normalization

        if logits.shape[1] == 1:
            # single logit, binary case
            logits = logits[:, 0]
            y = y.to(logits.dtype)
            loss_unreduced = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        else:
            loss_unreduced = F.cross_entropy(logits.float(), y.long(), reduction='none')
        return torch.dot(loss_unreduced, w)

    def loss_survival(self, scores, y):
        t = y[:, 0]  # times
        c = y[:, 1]  # censorships
        y_discrete = y[:, 1]  # bins
        # scores (batch_size, num_bins)
        return NLLSurvLoss.functional(scores, y_discrete, t, c)

    def loss(self, scores, logits, y, path):
        if self.task == 'classification':
            return self.loss_classification(logits, y, path)
        elif self.task == 'survival':
            return self.loss_survival(scores, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.init_lr)

    def update_warmup(self, batch_idx):
        return

    def training_step(self, batch, batch_idx):
        self.update_warmup(batch_idx)

        if isinstance(batch, list) and len(batch) == 1:
            batch = batch[0]

        x, y, path = batch[0], batch[2], batch[5]

        logits = self.forward(batch)
        loss = self.loss(logits, logits, y, path)  # bandaid fix TODO:

        self.logger.experiment.log_metric(self.logger.run_id, "train_loss", loss.detach().cpu())
        return loss

    def test_step(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 1:
            batch = batch[0]

        x, c, y, slide_uuid, patient_id, path = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]

        logits = self.forward(batch)

        return {'cohort': c.detach().cpu(), 'logits': logits.detach().cpu(), 'y': y.detach().cpu(),
                'slide_uuid': slide_uuid, 'patient_id': patient_id, 'path': path}

    @staticmethod
    def get_summary_df(outputs, task):
        if task == 'classification':
            return AbstractClassifier.get_summary_df_classification(outputs)
        elif task == 'survival':
            return AbstractClassifier.get_summary_df_survival(outputs)

    @staticmethod
    def get_summary_df_survival(outputs):
        path = np.concatenate([out["path"] for out in outputs])
        cohort = np.concatenate([out["cohort"] for out in outputs])
        slide_uuid = np.concatenate([out["slide_uuid"] for out in outputs])
        patient_id = np.concatenate([out["patient_id"] for out in outputs])

        survival_times = torch.concat([out["y"][:,0] for out in outputs]).numpy()
        censorship_flags = torch.concat([out["y"][:,1] for out in outputs]).numpy()


        bin_risk_logits = torch.concat([out["logits"] for out in outputs])

        risk_scores, _ = _calculate_risk(bin_risk_logits)



        df_res = pd.DataFrame({
            'risk': risk_scores,
            'survival': survival_times,
            'censorship': censorship_flags,
            "path": path,
            "slide_uuid": slide_uuid,
            "patient_id": patient_id,
            'cohort': cohort
        })

        df_patient = df_res.groupby(['patient_id',], as_index=False).agg({
            'survival': 'max',
            'censorship': 'max',
            'risk': 'mean',
            'cohort': 'max'
        })

        return df_patient

    @staticmethod
    def get_summary_df_classification(outputs):
        if len(outputs[-1]['logits'].shape) == 0:
            outputs[-1]['logits'] = outputs[-1]['logits'].unsqueeze(dim=0)

        logits = torch.concat([out["logits"] for out in outputs]).numpy()

        # binary case
        if len(logits.shape) == 2 and logits.shape[1] == 2:
            logits = logits[:, 1]

        y_true = torch.concat([out["y"] for out in outputs]).numpy()
        cohort = torch.concat([out["cohort"] for out in outputs]).numpy()
        path = np.concatenate([out["path"] for out in outputs])
        slide_uuid = np.concatenate([out["slide_uuid"] for out in outputs])
        patient_id = np.concatenate([out["patient_id"] for out in outputs])
        df = pd.DataFrame({
            "y_true": y_true,
            "cohort": cohort,
            "path": path,
            "slide_uuid": slide_uuid,
            "patient_id": patient_id,
        })
        if len(logits.shape) == 1:
            df['logits_0'] = logits
            df['y_pred'] = (logits > 0.5).astype(int)
        else:
            for dim in range(logits.shape[1]):
                df[f'logits_{dim}'] = logits[:, dim]
            y_pred = torch.argmax(torch.tensor(logits), dim=1)
            df['y_pred'] = y_pred

        # assumes num_classes=1 at this point
        # num_classes = 1 if len(logits.shape) == 1 else logits.shape[1]

        df_patient = df.groupby(['patient_id', 'cohort'], as_index=False).agg({
            'y_true': 'max',
            'logits_0': 'mean'
        })
        df_patient['y_pred'] = (df_patient.logits_0 > 0.5).astype(int)
        df_patient['logits'] = df_patient['logits_0']

        return df_patient

    @staticmethod
    def log_epoch_level_metrics(df_patient, task, logger):
        if task == 'survival':
            from sksurv.metrics import concordance_index_censored
            metrics = {}

            cindex, concordant, discordant, tied_risk, tied_time = concordance_index_censored(df_patient['censorship'].values == 0,
                                                                                              df_patient['survival'].values,
                                                                                              df_patient['risk'].values)
            metrics[f"all_c_index"] = cindex
            metrics[f"all_concordant"] = concordant
            metrics[f"all_discordant"] = discordant
            metrics[f"all_tied_risk"] = tied_risk
            metrics[f"all_tied_time"] = tied_time


            for cohort, df_cohort in df_patient.groupby('cohort'):
                cindex, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
                    df_cohort['censorship'].values == 0,
                    df_cohort['survival'].values,
                    df_cohort['risk'].values)
                metrics[f"{cohort}_c_index"] = cindex
                metrics[f"{cohort}_concordant"] = concordant
                metrics[f"{cohort}_discordant"] = discordant
                metrics[f"{cohort}_tied_risk"] = tied_risk
                metrics[f"{cohort}_tied_time"] = tied_time

            for m_str, m_value in metrics.items():
                logger.experiment.log_metric(logger.run_id, m_str, m_value)

            Logger.log(f'Metrics logged.', log_importance=1)
            Logger.log(metrics, log_importance=1)

            return metrics

        elif task == 'classification':
            metrics = {}
            suffix = '' # if num_classes == 1 else f'_{cls}'
            AbstractClassifier.add_patient_score(df_patient, 'accuracy', logger, metrics, suffix)
            AbstractClassifier.add_patient_score(df_patient, 'balanced_accuracy', logger, metrics, suffix)
            AbstractClassifier.add_patient_score(df_patient, 'f1', logger, metrics, suffix)
            AbstractClassifier.add_patient_score(df_patient, 'auc', logger, metrics, suffix)

            df_patient_cohort = df_patient.groupby('cohort').apply(lambda df_group: pd.Series(
                (accuracy_score(df_group.y_true, df_group.y_pred),
                balanced_accuracy_score(df_group.y_true, df_group.y_pred),
                 f1_score(df_group.y_true, df_group.y_pred),
                 calc_safe_auc(df_group.y_true, df_group.logits)),
                index=['accuracy', 'balanced_accuracy', 'f1', 'auc']
            ))

            for cohort, row in df_patient_cohort.iterrows():
                accuracy, balanced_accuracy, f1, auc = row['accuracy'], row['balanced_accuracy'], row['f1'], row['auc']
                logger.experiment.log_metric(logger.run_id, f"patient_C{cohort}_AUC{suffix}", auc)
                logger.experiment.log_metric(logger.run_id, f"patient_C{cohort}_Accuracy{suffix}", accuracy)
                logger.experiment.log_metric(logger.run_id, f"patient_C{cohort}_balanced_accuracy{suffix}", balanced_accuracy)
                logger.experiment.log_metric(logger.run_id, f"patient_C{cohort}_F1{suffix}", f1)
                metrics[f"patient_C{cohort}_AUC{suffix}"] = auc
                metrics[f"patient_C{cohort}_Accuracy{suffix}"] = accuracy
                metrics[f"patient_C{cohort}_balanced_accuracy{suffix}"] = balanced_accuracy
                metrics[f"patient_C{cohort}_F1{suffix}"] = f1

            Logger.log(f'Metrics logged.', log_importance=1)
            Logger.log(metrics, log_importance=1)

            return metrics

    @staticmethod
    def add_patient_score(df_patient, score, logger, metrics, suffix):
        if score.lower() == 'accuracy':
            patient_score = accuracy_score(df_patient.y_true, df_patient.y_pred)
        if score.lower() == 'balanced_accuracy':
            patient_score = balanced_accuracy_score(df_patient.y_true, df_patient.y_pred)
        elif score.lower() == 'f1':
            # patient_score = f1_score(df_patient.y_true, df_patient.y_pred)
            # optimal thresholding
            y_true = df_patient.y_true
            y_probs = df_patient.logits  # Ensure this is probabilities, not binary predictions
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
            # Step 3: Find the best threshold for F1 score
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            best_threshold = thresholds[f1_scores.argmax()]  # Threshold giving the highest F1
            # Step 4: Convert probabilities to class predictions using the best threshold
            y_pred_optimal = (y_probs >= best_threshold).astype(int)
            # Step 5: Compute F1 score
            patient_score = f1_score(y_true, y_pred_optimal)
        elif score.lower() == 'auc':
            patient_score = calc_safe_auc(df_patient.y_true, df_patient.logits)
        logger.experiment.log_metric(logger.run_id, f"patient_{score.upper()}{suffix}",
                                     patient_score)
        metrics[f"patient_{score.upper()}{suffix}"] = patient_score

