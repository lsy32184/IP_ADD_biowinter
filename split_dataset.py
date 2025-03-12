from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Dict, Literal, List


def split_data(
    dataframe: pd.DataFrame,
    stratify_col: List[str,],
    train_ratio: float,
    random_state: int,
) -> Dict[Literal["train", "val"], pd.DataFrame]:
    """
    Splits a dataframe into training and validation sets while ensuring stratified distribution.
    
    Args:
        dataframe (pd.DataFrame): The input dataframe containing the dataset.
        stratify_col (List[str]): Column(s) used for stratification.
        train_ratio (float): Proportion of data to allocate to the training set.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        Dict[Literal["train", "val"], pd.DataFrame]: Dictionary containing training and validation dataframes.
    """
    # Check of stratify cols are in the dataframe
    if isinstance(stratify_col, str):
        stratify_col = [stratify_col]

    for col in stratify_col:
        if col not in dataframe.columns:
            raise KeyError(f"Column {col} is not in the dataframe.")

    dataframe["strat_col"] = dataframe[stratify_col].apply(
        lambda row: "".join(row.astype(str)), axis=1
    )

    train_df, val_df = train_test_split(
        dataframe,
        test_size=1 - train_ratio,
        stratify=dataframe["strat_col"],
        shuffle=True,
        random_state=random_state,
    )

    return {"train": train_df, "val": val_df}


def test_split_data():
    """
    Tests the `split_data` function by visualizing label distributions before and after splitting.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    dataframe = pd.read_csv(r"D:\Soyeon\Project\metadata_R.csv")
    split = split_data(
        dataframe,
        stratify_col=["label_id", "source_id"],  #
        train_ratio=0.8,
        random_state=42,
    )

    print("train", split["train"].shape)
    print("val", split["val"].shape)

    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    sns.histplot(split["train"], x="label_id", ax=ax[0])
    sns.histplot(split["val"], x="label_id", ax=ax[1])
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    sns.histplot(split["train"], x="source_id", ax=ax[0])
    sns.histplot(split["val"], x="source_id", ax=ax[1])
    plt.show()


if __name__ == "__main__":
    test_split_data()
