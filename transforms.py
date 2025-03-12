"""
Python file to store transformations to train classification model
Assumption:
    1- The data received by the transformation is of type PIL Image
"""

import torchvision.transforms.v2 as v2
from torchvision.transforms.v2 import Transform
import torch
from typing import Tuple, Literal, Dict
from PIL import Image


def get_image_net_normalization_values() -> Dict[str, Tuple[float, float, float]]:
    # ImageNet normalization weights
    """
    Returns the mean and standard deviation values used for ImageNet normalization.

    Returns:
        Dict[str, Tuple[float, float, float]]: Dictionary containing `mean` and `std` tuples.
    """
    return {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}


def get_rotation_transformation() -> Transform:
    """
    Returns a transformation pipeline that applies random horizontal and vertical flips
    along with a random rotation of up to 30 degrees.

    Returns:
        Transform: A composed transformation pipeline.
    """
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=30),
        ]
    )


def get_float32_tensor_and_normalize_transformation(
    mean: Tuple[float, float, float] = None, std: Tuple[float, float, float] = None
) -> Transform:
    """
    Converts an image to a float32 tensor and normalizes it using the specified mean and standard deviation.

    Args:
        mean (Tuple[float, float, float], optional): Normalization mean values.
        std (Tuple[float, float, float], optional): Normalization standard deviation values.

    Returns:
        Transform: A composed transformation pipeline.
    """
    if mean is None or std is None:
        mean, std = tuple(get_image_net_normalization_values().values())

    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


def get_train_transformation(
    resize: int,
    scale: Tuple[float, float],
    norm_mean: Tuple[float, float, float],
    norm_std: Tuple[float, float, float],
) -> Transform:
    """
    Returns a transformation pipeline for training data, including resizing, cropping, rotation,
    color jittering, and normalization.

    Args:
        resize (int): The target image size after resizing.
        scale (Tuple[float, float]): Scale range for cropping.
        norm_mean (Tuple[float, float, float]): Normalization mean values.
        norm_std (Tuple[float, float, float]): Normalization standard deviation values.

    Returns:
        Transform: A composed transformation pipeline.
    """
    return v2.Compose(
        [
            v2.PILToTensor(),
            v2.RandomResizedCrop(
                size=resize,
                scale=scale if scale else (0.08, 1),
                interpolation=v2.InterpolationMode.BILINEAR,
            ),
            get_rotation_transformation(),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            get_float32_tensor_and_normalize_transformation(norm_mean, norm_std),
        ]
    )


def get_val_transformation(
    resize: int,
    norm_mean: Tuple[float, float, float],
    norm_std: Tuple[float, float, float],
) -> Transform:
    """
    Returns a transformation pipeline for validation data, including resizing and normalization.

    Args:
        resize (int): The target image size after resizing.
        norm_mean (Tuple[float, float, float]): Normalization mean values.
        norm_std (Tuple[float, float, float]): Normalization standard deviation values.

    Returns:
        Transform: A composed transformation pipeline.
    """
    return v2.Compose(
        [
            v2.PILToTensor(),
            v2.Resize(
                size=(resize, resize), interpolation=v2.InterpolationMode.BILINEAR
            ),
            get_float32_tensor_and_normalize_transformation(norm_mean, norm_std),
        ]
    )


def get_transform(
    split: Literal["train", "val"],
    resize: int,
    scale: Tuple[float, float] = None,
    norm_mean: Tuple[float, float, float] = None,
    norm_std: Tuple[float, float, float] = None,
) -> Transform:
    """
    Returns the appropriate transformation pipeline based on the dataset split (train or validation).
    
    Args:
        split (Literal["train", "val"]): The dataset split type.
        resize (int): The target image size after resizing.
        scale (Tuple[float, float]): Scale range for cropping.
        norm_mean (Tuple[float, float, float], optional): Normalization mean values.
        norm_std (Tuple[float, float, float], optional): Normalization standard deviation values.
    
    Returns:
        Transform: The appropriate transformation pipeline.
    """
    if split == "train":
        return get_train_transformation(resize, scale, norm_mean, norm_std)
    elif split == "val":
        return get_val_transformation(resize, norm_mean, norm_std)
    else:
        raise NotImplementedError(
            f"Split {split} is not implemented. Available 'train' and 'val'."
        )


def test_get_transform() -> None:
    mock_data = torch.randint(0, 255, size=(3, 512, 512))
    for split in ["train", "val"]:
        transform = get_transform(split, resize=224, scale=(0.8, 1))

        # Apply transform to mock data
        mock_data_transformed = transform(mock_data)

        print(
            f"SPLIT {split} ~~ Before transform : {mock_data.shape=}, {mock_data.dtype=}"
        )
        print(
            f"SPLIT {split} ~~ After transform : {mock_data_transformed.shape=}, {mock_data_transformed.dtype=}"
        )


def visualize_transform(
    data: Image,
    resize: int,
    scale: Tuple[float, float],
    norm_mean: Tuple[float, float, float] = None,
    norm_std: Tuple[float, float, float] = None,
) -> None:
    import matplotlib.pyplot as plt

    def denormalize_to_uint8(data, mean=None, std=None):
        if mean is None or std is None:
            mean, std = tuple(get_image_net_normalization_values().values())

        data = (data * torch.tensor(std)) + torch.tensor(mean)  # Reverse normalization
        data = (data - torch.min(data)) / (torch.max(data) - torch.min(data)) * 255
        return torch.clip(data, 0, 255).to(torch.uint8)

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for i, split in enumerate(["train", "val"]):
        transform = get_transform(split, resize, scale, norm_mean, norm_std)
        transformed_data = transform(data)

        ax[i, 0].imshow(data)
        ax[i, 1].imshow(denormalize_to_uint8(transformed_data.permute(1, 2, 0)))
        ax[i, 0].set_title(f"SPLIT {split} Before transform, {data.size}")
        ax[i, 1].set_title(f"SPLIT {split} After transform {transformed_data.shape}")
    plt.show()


def test_visualize_transform():
    import pandas as pd
    import random

    df = pd.read_csv(r"D:\Soyeon\Project\metadata_R.csv")

    # Load image
    image = Image.open(df.iloc[random.randint(0, len(df) - 1)].image_path).convert(
        "RGB"
    )

    visualize_transform(
        image,
        resize=224,
        scale=(0.8, 1),
    )


if __name__ == "__main__":
    test_get_transform()
    test_visualize_transform()
