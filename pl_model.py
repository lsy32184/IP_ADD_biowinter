import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim.swa_utils import AveragedModel
from metrics import compute_metrics
import copy
import numpy as np
import wandb


class Classifier(pl.LightningModule):
    """
    A PyTorch Lightning classifier supporting various architectures, learning rate scheduling,
    and optional advanced training techniques.
    
    Args:
        model_name (str): The backbone model to use ("resnet", "inception", "vit-s14").
        num_classes (int): Number of target classes.
        pretrained (bool): Whether to load a pretrained model.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for regularization.
        optimizer (str): Optimizer type ("AdamW", "Adam", "SGD").
        use_layer_freezing (bool, optional): Whether to freeze lower layers.
        use_lora_adapter (bool, optional): Whether to use LoRA adapters.
        use_gradient_clipping (bool, optional): Apply gradient clipping.
        label_smoothing (float): Label smoothing value.
        ema_decay (float, optional): EMA decay factor.
        epochs (int): Number of training epochs.
    """
    def __init__(
        self,
        model_name="inception",
        num_classes=4,
        pretrained=True,
        learning_rate=1e-3,
        weight_decay=1e-4,
        optimizer="AdamW",
        use_layer_freezing=None,
        use_lora_adapter=False,
        use_gradient_clipping=False,
        label_smoothing=0,
        ema_decay=None,
        epochs=20,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.epochs = epochs
        self.use_gradient_clipping = use_gradient_clipping
        self.ema_decay = ema_decay
        self.ema = None

        # 모델 로드
        if model_name == "resnet":
            self.model = models.resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        elif model_name == "inception":
            self.model = models.inception_v3(pretrained=pretrained, aux_logits=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        elif model_name == "vit-s14":
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.classifier_head = nn.Linear(384, num_classes)

            if use_layer_freezing:
                for name, param in self.backbone.named_parameters():
                    if not ("blocks.11" in name or "blocks.12" in name):
                        param.requires_grad = False

                if use_lora_adapter:
                    from peft import LoraConfig, get_peft_model

                    lora_config = LoraConfig(
                        r=16,
                        lora_alpha=32,
                        lora_dropout=0.1,
                        bias="none",
                        task_type="SEQ_CLS",
                        target_modules=["qkv", "query", "key", "value"],
                    )
                    self.backbone = get_peft_model(self.backbone, lora_config)

        # 손실 함수 정의
        self.criterion = (
            nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            if label_smoothing > 0
            else nn.CrossEntropyLoss()
        )

        # 하이퍼파라미터 저장
        self.save_hyperparameters()

        # 예측값 저장 리스트 추가
        self.train_preds, self.train_labels, self.train_losses = [], [], []
        self.val_preds, self.val_labels, self.val_losses = [], [], []

    def _ema_avg_fn(self, avg_p, p):
        return self.ema_decay * avg_p + (1 - self.ema_decay) * p

    def on_train_epoch_start(self):
        if self.ema_decay is not None and self.ema is None:
            self.ema = AveragedModel(copy.deepcopy(self), avg_fn=self._ema_avg_fn).to(
                self.device
            )

    def forward(self, x):
        """
        Performs a forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Model output logits.
        """
        if self.model_name == "vit-s14":
            if not self.training:
                with torch.no_grad():
                    features = self.backbone(x)
            else:
                features = self.backbone(x)

            cls_token = features[:, 0, :]
            logits = self.classifier_head(cls_token)
            return logits

        elif self.model_name == "inception":
            outputs = self.model(x)
            return outputs[0] if isinstance(outputs, tuple) else outputs
        else:
            return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Defines the training step, computing loss and logging metrics.
        
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of images and labels.
            batch_idx (int): Batch index.
        
        Returns:
            torch.Tensor: Computed loss value.
        """
        image, target = batch
        image, target = image.to(self.device), target.to(self.device)

        outputs = self(image)  
        loss = self.criterion(outputs, target)

        self.train_losses.append(loss.item())
        self.train_preds.append(
            torch.argmax(outputs, dim=1).cpu().numpy()
        ) 
        self.train_labels.append(torch.argmax(target, dim=1).cpu().numpy())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step, computing loss and logging validation metrics.
        
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of images and labels.
            batch_idx (int): Batch index.
        
        Returns:
            torch.Tensor: Computed validation loss.
        """
        image, target = batch

        image, target = image.to(self.device), target.to(self.device)

        outputs = self(image)
        loss = self.criterion(outputs, target)

        self.val_losses.append(loss.item())
        self.val_preds.append(
            torch.argmax(outputs, dim=1).cpu().numpy()
        ) 
        self.val_labels.append(torch.argmax(target, dim=1).cpu().numpy())
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.ema is not None:
            for ema_param, param in zip(self.ema.parameters(), self.parameters()):
                ema_param.data = self._ema_avg_fn(ema_param.data, param.data)

        if len(self.train_losses) > 0:
            avg_train_loss = np.mean(self.train_losses)

            train_preds = np.concatenate(self.train_preds, axis=0)
            train_labels = np.concatenate(self.train_labels, axis=0)

            metrics_dict = compute_metrics(train_labels, train_preds)

            # Train Loss & Metrics 로깅
            self.log("train_loss", avg_train_loss, prog_bar=True)
            self.log("train_accuracy", metrics_dict["accuracy"], prog_bar=True)
            self.log("train_precision", metrics_dict["precision"])
            self.log("train_recall", metrics_dict["recall"])
            self.log("train_f1_score", metrics_dict["f1_score"], prog_bar=True)

            wandb.log(
                {
                    "train_loss": avg_train_loss,
                    "train_accuracy": metrics_dict["accuracy"],
                    "train_f1_score": metrics_dict["f1_score"],
                    "train_recall": metrics_dict["recall"],
                    "train_precision": metrics_dict["precision"],
                }
            )

            # 리스트 초기화
            self.train_losses.clear()
            self.train_preds.clear()
            self.train_labels.clear()

    def on_validation_epoch_end(self):
        avg_loss = np.mean(self.val_losses)

        val_preds = np.concatenate(self.val_preds, axis=0)
        val_labels = np.concatenate(self.val_labels, axis=0)

        metrics_dict = compute_metrics(val_labels, val_preds)

        # Validation Loss & Metrics 로깅
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_accuracy", metrics_dict["accuracy"], prog_bar=True)
        self.log("val_precision", metrics_dict["precision"])
        self.log("val_recall", metrics_dict["recall"])
        self.log("val_f1_score", metrics_dict["f1_score"], prog_bar=True)

        wandb.log(
            {
                "val_loss": avg_loss,
                "val_accuracy": metrics_dict["accuracy"],
                "val_f1_score": metrics_dict["f1_score"],
                "val_recall": metrics_dict["recall"],
                "val_precision": metrics_dict["precision"],
            }
        )

        # 리스트 초기화
        self.val_losses.clear()
        self.val_preds.clear()
        self.val_labels.clear()

    def configure_optimizers(self):
        """
        Configures the optimizer and optionally applies gradient clipping.
        
        Returns:
            dict: Optimizer configuration.
        """
        optimizers = {
            "AdamW": torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            ),
            "Adam": torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            ),
            "SGD": torch.optim.SGD(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            ),
        }

        optimizer = optimizers.get(
            self.optimizer,
            torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            ),
        )

        if self.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)

        return {"optimizer": optimizer}


def test_classifier():
    """
    Tests the classifier by running a sample forward pass with a mock dataset.
    """
    from setup_data import get_sample_training_loader

    model = Classifier(
        model_name="inception",
        num_classes=4,
        pretrained=True,
        learning_rate=1e-3,
        weight_decay=1e-4,
        optimizer="AdamW",
        use_layer_freezing=None,
        use_lora_adapter=False,
        use_gradient_clipping=False,
        label_smoothing=0,
        ema_decay=None,
        epochs=20,
    )

    # test training step and val step
    loaders = get_sample_training_loader()
    for split, loader in loaders.items():
        for i, batch in enumerate(loader):
            with torch.no_grad():
                if split == "train":
                    # training step
                    loss = model.training_step(batch, i)

                else:
                    loss = model.validation_step(batch, i)
                print(loss)
                break


if __name__ == "__main__":
    test_classifier()
