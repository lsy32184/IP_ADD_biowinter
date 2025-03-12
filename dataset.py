import os


import torch
import torch.nn as nn

from PIL import Image

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir=None, transform=None):
        self.root_dir = root_dir
        dataframe.image_path = dataframe.image_path.apply(
            lambda x: os.path.join(self.root_dir, os.path.basename(x))
        )
        self.data = dataframe[["image_path", "label_id"]].to_dict(orient="records")
        self.transform = transform
        self.num_classes = dataframe["label_id"].nunique()

        # Assign dir name to image path (allow changing disk location)

    def __getitem__(self, idx):
        image_path = self.data[idx]["image_path"]

        image = Image.open(image_path).convert("RGB")

        label = torch.tensor(self.data[idx]["label_id"])

        # Always one hot encoding!
        # Reason : since there is a chance cutmix or mixup doesnt occur,
        # to ensure data consistency all labels are one hot encoded

        label = nn.functional.one_hot(
            label, num_classes=self.num_classes
        ).to(
            torch.float32  # make sure classes are float to ensure consistency on cutmix/mixup
        )

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data)


"""
Dataset : load 1 data point (image, label), length (total # datapoints), __getitem__ to retrieve data using indexing (with or without transforming it)

Dataloader : batch + shuffle the data, the way it works : randomly pick with NO replacement indices from (0, len(dataset) - 1) to fill up a batch of batch size 

Cutmix/Mixup : after batching. 

Separate concerns !!! 

"""


def test_customdataset():
    from transforms import get_transform
    import pandas as pd

    df = pd.read_csv(r"D:\Soyeon\Project\metadata_R.csv")
    root = r"C:\\Users\\BDA_INT01\\Desktop\\Soyeon Lee\\Project\\MARCO_training"

    transform = get_transform(split="train", resize=299, scale=(0.8, 1, 0))
    dataset = CustomDataset(df, root, transform)

    assert dataset.num_classes == 4, "Wrong number of classes, expected 4"
    assert len(dataset) == len(df), (
        "Incorrect match between dataset length and dataframe length"
    )

    image, label = dataset[0]

    # test apply transform
    assert (image.shape[1] == 299) and (image.shape[2] == 299), (
        "Incorrect transform apply. Expected shape H=299, W=299."
    )

    # test one-hot encoded
    assert label.shape[0] == dataset.num_classes, "Wrong one hot encoding."

    assert torch.sum(label) == 1, f"Issue with one-hot encoding {label}"

    return dataset


if __name__ == "__main__":
    test_customdataset()
