import os


import torch
import torch.nn as nn

from PIL import Image

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading image classification data.
    
    Args:
        dataframe (pd.DataFrame): Dataframe containing image paths and label IDs.
        root_dir (str, optional): Root directory where image files are stored.
        transform (callable, optional): Transformations to be applied to the images.
    
    Attributes:
        root_dir (str): Root directory for image files.
        data (List[Dict]): List of dictionaries containing `image_path` and `label_id`.
        transform (callable): Transformation function.
        num_classes (int): Number of unique classes in the dataset.
    """
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
        """
        Retrieves an image and its corresponding label.
        
        Args:
            idx (int): Index of the image-label pair to be retrieved.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image and one-hot encoded label.
        """
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
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)


"""
Dataset : load 1 data point (image, label), length (total # datapoints), __getitem__ to retrieve data using indexing (with or without transforming it)

Dataloader : batch + shuffle the data, the way it works : randomly pick with NO replacement indices from (0, len(dataset) - 1) to fill up a batch of batch size 

Cutmix/Mixup : after batching. 

Separate concerns !!! 

"""


def test_customdataset():
    """
    Function to test the functionality of the `CustomDataset` class.
    
    Steps:
        1. Loads the dataset from a CSV file.
        2. Applies transformations.
        3. Checks dataset properties such as class count and length.
        4. Verifies image transformation and shape.
        5. Ensures labels are correctly one-hot encoded.
    
    Returns:
        CustomDataset: The dataset instance.
    """
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
