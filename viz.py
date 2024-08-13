import os
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from rf_utils.vision import get_keypoints_from_pred


@dataclass
class Line:
    """Dataclass to represent a line"""
    xy_from: tuple[int, int]
    xy_to: tuple[int, int]
    color: tuple[int, int, int]
    thickness: int


def img_with_lines(image: np.ndarray, lines: list[Line]) -> np.ndarray:
    """Paints the lines on the image, without modifying the original image

    Args:
        image: The image
        lines: The array of lines to be drawn

    Returns:
        The image with lines
    """
    return_img = image.copy()
    for line in lines:
        return_img = cv2.line(return_img, line.xy_from,
                              line.xy_to, line.color, line.thickness)
    return return_img


def img_with_lines_from_pred(image: np.ndarray, pred) -> np.ndarray:
    """Extracts kps and paints corresponding lines on the image

    Args:
        image: The image
        pred: The raw or softmaxed prediction from the model

    Returns:
        Image with lines
    """
    vp_coords, ll_coords, lr_coords = get_keypoints_from_pred(pred, factor=4)[0]
    vp_coords, ll_coords, lr_coords = vp_coords.tolist(), ll_coords.tolist(), lr_coords.tolist()
    return img_with_lines_from_kps(image, vp_coords, ll_coords, lr_coords)


def img_with_lines_from_kps(image, vp, ll, lr) -> np.ndarray:
    """Draws lines corresponding to the keypoints on the image

    Args:
        image: The image
        vp: The vanishing point
        ll: The left intersect with x-axis
        lr: The right intersect with x-axis

    Returns:
        Image with lines
    """
    image_with_lines = cv2.circle(
        image, vp, 5, (255, 0, 0), -1)

    lines = [Line(vp, ll, (0, 255, 0), 2),
             Line(vp, lr, (0, 0, 255), 2)]

    image_with_lines = img_with_lines(
        image_with_lines, lines)

    return image_with_lines


class PlotInfo:
    """Represents the information needed to plot an array
    """

    def __init__(self, array, isImage, vmin=None, vmax=None, gradientColor="gray", title=None):
        # TODO: Add type hints, maybe restructure to dataclass
        self.array = array
        self.isImage = isImage
        self.gradientColor = gradientColor
        self.title = title

        self.vmin = vmin
        if self.vmin is None:
            self.vmin = array.min()
        self.vmax = vmax
        if self.vmax is None:
            self.vmax = array.max()


def plot_multiple_color_maps(*arrays_of_plotinfos: list[PlotInfo], full_title: str = "title", dir_path: str = "output", base_filename: str = "plot") -> None:
    """Plots multiple color maps or images in a single figure

    Args:
        arrays_of_plotinfos: The arrays of maps or images to be plotted
        full_title: The title displayed at the top. Defaults to "title".
        dir_path: The path to the output directory. Defaults to "output".
        base_filename: The filename of the output file, without file format. Defaults to "plot".
    """
    n_columns = len(arrays_of_plotinfos)  # one column for each array
    n_rows = len(arrays_of_plotinfos[0])  # assuming each array has same length
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(5 * n_columns, 3 * n_rows))

    # Plot arrays for each color
    for row in range(n_rows):
        for column in range(n_columns):
            plotinfo = arrays_of_plotinfos[column][row]
            if plotinfo.isImage:
                axes[row, column].imshow(plotinfo.array)
            else:
                axes[row, column].imshow(plotinfo.array, cmap=mcolors.LinearSegmentedColormap.from_list(
                    "", ["white", plotinfo.gradientColor]), vmin=plotinfo.vmin, vmax=plotinfo.vmax)
            title = plotinfo.title or f"{row}, {column}"
            axes[row, column].set_title(title)

    plt.tight_layout(pad=3.0)
    # Adjust the top to make space for the suptitle
    plt.subplots_adjust(top=0.97)
    fig.suptitle(full_title, fontsize=16)

    i = 0
    ext = ".pdf"
    filename = f"{base_filename}{ext}"
    os.makedirs(dir_path, exist_ok=True)

    filepath = os.path.join(dir_path, filename)

    # Check if the file exists and create a new filename if necessary
    while os.path.exists(filepath):
        i += 1
        filename = f"{base_filename}_{i}{ext}"
        filepath = os.path.join(dir_path, filename)

    plt.savefig(filepath)
