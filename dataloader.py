from torch.utils.data import DataLoader, default_collate
from typing import List
from functools import partial
import torch
import torchvision.transforms.v2 as v2
from dataset import CustomDataset


def batch_augment_collate(
    batch: List[torch.Tensor], alpha: float, augment_prob: float, cutmix_prob: float
):
    # default collate, [(C, H, W), ...., (C, H, W)] with number of element in the list equal to the batch size ---> (Batch_size, C, H, W)
    # default collate, [(,num_classes ), ..., (, num_classes)] with number of element in the list equal to the batch size ---> (Batch_size, num_classes) "vector"
    """
    Collate function that applies CutMix or MixUp augmentations to the batch before returning it.

    Args:
        batch (List[torch.Tensor]): A list of tensors representing a batch of images and labels.
        alpha (float): The hyperparameter for the Beta distribution controlling the interpolation strength.
        augment_prob (float): The probability of applying augmentation.
        cutmix_prob (float): The probability of using CutMix instead of MixUp when augmentation is applied.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Augmented batch of images and labels.
    """
    return apply_cutmix_mixup(default_collate(batch), alpha, augment_prob, cutmix_prob)


def apply_cutmix_mixup(batch, alpha, augment_prob, cutmix_prob):
    # !!!!! This function assumes labels are ONE HOT encoded already !!!!!
    """
    Applies CutMix or MixUp augmentations to the batch based on the provided probabilities.

    Args:
        batch (Tuple[torch.Tensor, torch.Tensor]): A tuple of images and labels tensors.
        alpha (float): The Beta distribution parameter for generating lambda values.
        augment_prob (float): Probability of applying augmentation.
        cutmix_prob (float): Probability of using CutMix over MixUp.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Augmented images and labels if augmentation is applied, otherwise returns original batch.
    """
    images, labels = batch
    if torch.rand(1).item() < augment_prob:
        if torch.rand(1).item() < cutmix_prob:
            cutmix = v2.CutMix(num_classes=None, alpha=alpha)
            return cutmix(images, labels)
        else:
            mixup = v2.MixUp(num_classes=None, alpha=alpha)
            return mixup(images, labels)
    return images, labels


def get_dataloader(
    dataset: CustomDataset,
    batch_size,
    num_workers,
    shuffle: bool,
    cutmix_prob,
    alpha,
    augment_prob,
):
    """
    Creates a PyTorch DataLoader with optional CutMix and MixUp augmentation support.

    Args:
        dataset (CustomDataset): The dataset to load.
        batch_size (int, optional): Number of samples per batch. Default is `32`.
        use_batch_augmentation (bool) complete here
        num_workers (int, optional): Number of worker threads for data loading. Default is `4`.
        cutmix_prob (float, optional): Probability of using CutMix over MixUp. Default is `0.5`.
        alpha (float, optional): Beta distribution parameter for mixing images. Default is `1.0`.
        augment_prob (float, optional): Probability of applying augmentation. Default is `0.5`.

    Returns:
        torch.utils.data.DataLoader: The configured DataLoader instance.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=num_workers > 0,
        collate_fn=partial(
            batch_augment_collate,
            alpha=alpha,
            augment_prob=augment_prob,
            cutmix_prob=cutmix_prob,
        )
        if augment_prob > 0
        else default_collate,
    )
    return loader


"""
in main
~~~ Train Setup ~~~
load df, 
split df
create dataset for each split
create dataloader for each split

"""


def test_dataloader():
    """
    Function to test the DataLoader setup by loading a sample batch and checking its shape and label properties.

    Steps:
        1. Loads a dataset from a CSV file.
        2. Applies a transformation using `get_transform()`.
        3. Creates a `CustomDataset` instance.
        4. Initializes a DataLoader using `get_dataloader()`.
        5. Fetches a batch and prints its shape.
        6. Verifies that labels are correctly one-hot encoded.

    Assertions:
        - Sum of one-hot vector labels must be `1`.
        - Labels must be of `torch.float32` type.
    """
    from dataset import CustomDataset
    from transforms import get_transform
    import pandas as pd
    from utils import visualize_batch

    batch_size = 32
    num_workers = 0
    cutmix_prob = 0.5
    alpha = 1.0
    augment_prob = 0.5
    shuffle = True
    df = pd.read_csv(r"D:\Soyeon\Project\metadata_R.csv")
    root = r"C:\\Users\\BDA_INT01\\Desktop\\Soyeon Lee\\Project\\MARCO_training"

    transform = get_transform(split="train", resize=299, scale=(0.8, 1, 0))
    dataset = CustomDataset(df, root, transform)

    dataloader = get_dataloader(
        dataset,
        batch_size,
        num_workers,
        shuffle,
        cutmix_prob,
        alpha,
        augment_prob,
    )
    # get a batch of item

    for i, batch in enumerate(dataloader):
        image, label = batch
        print(image.shape)
        print(label.shape)
        assert torch.all(torch.sum(label, dim=1) == torch.ones(batch_size)), (
            "Sum of one-hot vector must be 1."
        )
        assert label.dtype == torch.float32, (
            "Wrong dtype for label, expected torch.float32."
        )
        visualize_batch(image, fig_title="sampled batch")
        break


if __name__ == "__main__":
    test_dataloader()
