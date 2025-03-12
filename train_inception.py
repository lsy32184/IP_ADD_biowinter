import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import wandb
from setup_data import setup_training_loaders
from pl_model import Classifier

torch.set_float32_matmul_precision("high")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# inception 하이퍼파라미터 탐색 설정
sweep_config_inception = {
    "method": "bayes",
    "metric": {"name": "val_f1_score", "goal": "maximize"},
    "parameters": {
        "model_name": {"values": ["inception"]},
        "learning_rate": {"values": [0.0019763381137381368]},
        "weight_decay": {"values": [0.01]},
        "batch_size": {"values": [128]},  # 128 good
        "epochs": {"values": [30]},
        "optimizer": {"values": ["AdamW"]},
        "label_smoothing": {"values": [0.0]},
        "ema_decay": {"values": [0.99]},
        "num_workers": {"values": [12]},
    },
}


def main():
    """
    Initializes the training process for the Inception model.
    
    Steps:
        1. Loads dataset and applies transformations.
        2. Configures the model and optimizer.
        3. Sets up training with early stopping and model checkpointing.
        4. Trains the model using PyTorch Lightning.
    """
    csv_path = r"C:\\Users\\BDA_INT01\\Desktop\\Soyeon Lee\\Project\\metadata_R.csv"
    root_dir = r"C:\Users\BDA_INT01\Documents\MARCO"
    wandb.init(project="inception_experiment")
    config = wandb.config

    model_name = config["model_name"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    optimizer = config["optimizer"]
    label_smoothing = config["label_smoothing"]
    ema_decay = config["ema_decay"]
    num_workers = config["num_workers"]

    loaders = setup_training_loaders(
        path=csv_path,
        image_dir=root_dir,
        train_ratio=0.8,
        stratify_col="label_id",
        image_resize=299,
        scale_random_crop=(0.8, 1),
        batch_size=batch_size,
        num_workers=num_workers,
        cutmix_prob=0.5,
        alpha=1.0,
        batch_augment_prob=0.5,
        random_state=42,
    )

    model = Classifier(
        model_name=model_name,
        num_classes=4,
        pretrained=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_layer_freezing=False,
        use_lora_adapter=False,
        label_smoothing=label_smoothing,
        ema_decay=ema_decay,
        epochs=epochs,
        optimizer=optimizer,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=DEVICE,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[
            EarlyStopping(monitor="val_f1_score", mode="max", patience=5),
            ModelCheckpoint(
                monitor="val_f1_score",
                mode="max",
                save_top_k=-1,
                dirpath="checkpoints/",
            ),
        ],
    )

    trainer.fit(model, loaders["train"], loaders["val"])


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_config_inception, project="inception_experiment")
    wandb.agent(sweep_id, function=main, count=1)
