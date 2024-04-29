import numpy as np
from typing import Optional


def get_keypoints_from_pred(pred: np.ndarray, factor: int = 1) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Returns the keypoints from the prediction

    Args:
        pred: Prediction from the model. Expected shape (3, H, W)
        factor: Extracted keypoints are multiplied with this. Defaults to 1.

    Returns:
        Tuple of keypoints (vp, ll, lr)
    """
    vp_y, vp_x = np.unravel_index(pred[0].argmax(), pred[0].shape)
    ll_y, ll_x = np.unravel_index(pred[1].argmax(), pred[1].shape)
    lr_y, lr_x = np.unravel_index(pred[2].argmax(), pred[2].shape)

    return (vp_x * factor, vp_y * factor), (ll_x * factor, ll_y * factor), (lr_x * factor, lr_y * factor)


def min_max_norm(img: np.ndarray) -> np.ndarray:
    """Performs min-max normalization on an image. Expects image in HxWxC format

    Args:
        img: Img to be min_max normalized

    Returns:
         Normalized image
    """
    return (img - img.min(axis=(0, 1))) / (img.max(axis=(0, 1)) - img.min(axis=(0, 1)))


def find_intersection(line1: tuple[tuple[int, int], tuple[int, int]], line2: tuple[tuple[int, int], tuple[int, int]]) -> Optional[tuple[int, int]]:
    """Finds the intersection point of two lines

    Args:
        line1: The first line
        line2: The second line

    Returns:
        The coordinates of the intersection lines
    """
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denominator == 0:
        # Lines are parallel, no intersection point
        return None

    intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) -
                      (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) -
                      (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    return (int(intersection_x), int(intersection_y))


def get_coordinates_on_frame(vp: tuple[int, int], kp: tuple[int, int], dim: tuple[int, int] = (319, 223)) -> tuple[int, int]:
    """Returns the coordinates of the kp projected on the lower half of the frame

    Args:
        vp: The vanishing points
        kp: The keypoint to be projected
        dim: The dimension of the frame as (width, height). Note that this takes 0 indexing into account. Defaults to (319, 223).

    Returns:
        The projected keypoint
    """
    line1 = (vp, kp)
    line2 = ((0, dim[1]), (dim[0], dim[1]))
    intersect = find_intersection(line1, line2)
    if intersect == None:
        return kp

    if intersect[0] > dim[0] or intersect[0] < 0:
        if intersect[0] > dim[0]:
            x = dim[0]
        else:
            x = 0
        line2 = ((x, 0), (x, dim[1]))
        intersect = find_intersection(line1, line2)

    if intersect == None:
        return kp

    return intersect
