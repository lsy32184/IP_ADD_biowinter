import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import math
from typing import Tuple


def visualize_batch(
    images: torch.Tensor,
    fig_title: str,
    nrow: int = None,
    figsize: Tuple[int, int] = (10, 10),
) -> None:
    """
    Visualizes a batch of images in a grid layout using matplotlib.

    Args:
        images (torch.Tensor): A batch of images of shape (B, C, H, W).
        fig_title (str): Figure title.
        nrow (int, optional): Number of images per row in the grid. Default is sqrt(B).
        figsize (tuple, optional): Figure size for matplotlib.
    """
    # Ensure the input is a tensor
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    else:
        raise TypeError("Expected images to be a torch.Tensor")

    B = images.shape[0]
    if nrow is None:
        nrow = math.ceil(math.sqrt(B))  # Approximate square layout

    grid = vutils.make_grid(images, nrow=nrow, normalize=True, scale_each=True)

    # Convert tensor to numpy for matplotlib
    grid = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=figsize)
    plt.title(fig_title)
    plt.imshow(grid)
    plt.axis("off")
    plt.show()
