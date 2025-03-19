from pytorch_lightning.callbacks import Callback
from src.components.models.AbstractClassifier import AbstractClassifier
from datetime import datetime
from src.components.objects.Logger import Logger


class SaveAndLogOutputsCallback(Callback):
    METRICS = []

    def __init__(self, path, task, logger):
        self.path = path
        self.task = task
        self.logger = logger
        self.outputs = []

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self.outputs.append(outputs)

    def on_test_end(self, trainer, pl_module, *args, **kwargs):
        df_patient_pred = AbstractClassifier.get_summary_df(self.outputs, self.task)
        if self.logger is not None:
            metrics = AbstractClassifier.log_epoch_level_metrics(df_patient_pred, self.task, self.logger)
            SaveAndLogOutputsCallback.METRICS.append(metrics)
        if self.path is not None:
            time_str = datetime.now().strftime('%d_%m_%Y_%H_%M')
            df_patient_pred.to_csv(self.path.format(time=time_str), index=False)
            Logger.log(f"Outputs saved in {self.path.format(time=time_str)}", log_importance=1)
        self.outputs = []  # to save memory


