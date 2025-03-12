from split_dataset import split_data
from dataset import CustomDataset
from dataloader import get_dataloader
from transforms import get_transform

from typing import Dict, Literal, Tuple, List
from torch.utils.data import DataLoader
import pandas as pd

"""
dataloading :

    path_metadata : mypath
    root_dir_img : mydir
    random_resize_crop : 299
    scale_random_resize_crop : (0.8, 1)
    use_cutmix : True
    cutmix_prob : 0.5,
    alpha : 1.0,
    batch_augment_prob : 0.5, 

"""


def setup_training_loaders(
    path: str,
    image_dir: str,
    train_ratio: float,
    stratify_col: List[str,],
    image_resize: int,
    scale_random_crop: Tuple[float, float],
    batch_size: int,
    num_workers: int,
    cutmix_prob: float,
    alpha: float,
    batch_augment_prob: float,
    random_state: int,
) -> Dict[Literal["train", "val"], DataLoader]:
    """
    Sets up data loaders for training and validation.
    
    Args:
        path (str): Path to the dataset metadata CSV file.
        image_dir (str): Directory containing image files.
        train_ratio (float): Proportion of data used for training.
        stratify_col (List[str]): Column(s) to stratify the data split.
        image_resize (int): Target image size after resizing.
        scale_random_crop (Tuple[float, float]): Scale range for random cropping.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker threads for data loading.
        cutmix_prob (float): Probability of applying CutMix augmentation.
        alpha (float): MixUp alpha value for augmentation.
        batch_augment_prob (float): Probability of applying batch augmentation.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        Dict[Literal["train", "val"], DataLoader]: A dictionary containing data loaders for training and validation.
    """
    dataframe = pd.read_csv(path)

    splits = split_data(dataframe, stratify_col, train_ratio, random_state)

    loaders = {}

    for split, df in splits.items():
        transform = get_transform(split, image_resize, scale=scale_random_crop)
        dataset = CustomDataset(df, image_dir, transform)
        loaders[split] = get_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True if split == "train" else False,
            cutmix_prob=cutmix_prob,
            alpha=alpha,
            augment_prob=batch_augment_prob,
        )

    return loaders


def setup_test_loader():
    """
    Placeholder function for setting up the test data loader.
    """
    pass


def get_sample_training_loader():
    """
    Returns sample training and validation data loaders using predefined parameters.
    
    Returns:
        Dict[Literal["train", "val"], DataLoader]: Sample data loaders for training and validation.
    """

    path_metadata = r"D:\Soyeon\Project\metadata_R.csv"
    image_dir = r"C:\\Users\\BDA_INT01\\Desktop\\Soyeon Lee\\Project\\MARCO_training"
    train_ratio = 0.8
    stratify_col = "label_id"
    image_resize = 299
    scale_random_crop = (0.8, 1)
    batch_size = 64
    num_workers = 0
    cutmix_prob = 0.1
    alpha = 1.0
    batch_augment_prob = 1
    random_state = 42

    loaders = setup_training_loaders(
        path=path_metadata,
        image_dir=image_dir,
        train_ratio=train_ratio,
        stratify_col=stratify_col,
        image_resize=image_resize,
        scale_random_crop=scale_random_crop,
        batch_size=batch_size,
        num_workers=num_workers,
        cutmix_prob=cutmix_prob,
        alpha=alpha,
        batch_augment_prob=batch_augment_prob,
        random_state=random_state,
    )
    return loaders


def test_setup_training_loaders():
    """
    Tests the data loader setup by visualizing a sample batch from training and validation sets.
    """
    from utils import visualize_batch

    loaders = get_sample_training_loader()

    for split, loader in loaders.items():
        for batch in loader:
            image, label = batch
            visualize_batch(image, fig_title=f"Split {split}")
            break


if __name__ == "__main__":
    test_setup_training_loaders()
