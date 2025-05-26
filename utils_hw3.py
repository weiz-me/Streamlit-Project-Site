from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def get_data_path(filename: Path) -> str:
    """
    Helper function to teturn the path to a data file.

    Args:
        filename (Path): The filename.

    Returns:
        str: Absolute path to the file.
    """
    return str((DATA_DIR / filename).resolve())


def get_result_path(filename: Path) -> str:
    """
    Helper function to teturn the path to a result file.

    Args:
        filename (Path): The filename.

    Returns:
        str: Absolute path to the file.
    """
    return str((RESULTS_DIR / filename).resolve())


def imread(
    path: Path, flag: int = cv2.IMREAD_COLOR, rgb: bool = False, normalize: bool = False
) -> np.ndarray:
    """
    Reads an image from file.

    Args:
        path (Path): Image path.
        flag (int, optional): cv2.imread flag. Defaults to cv2.IMREAD_COLOR.
        rgb (bool, optional): Convert BGR to RGB. Defaults to False.
        normalize (bool, optional): Normalize values to [0, 1]. Defaults to False.

    Raises:
        FileNotFoundError: File does not exist in the specified location.

    Returns:
        np.ndarray: Loaded image.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"File not found: {path}")
    img = cv2.imread(str(path), flag)

    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if normalize:
        img = img.astype(np.float64) / 255
    return img


def imread_alpha(path: Path, normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads an image containing an alpha channel from file. The alpha channel is returned separately.

    Args:
        path (Path): Image path.
        normalize (bool, optional): Normalizae values to [0, 1]. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The BGR image and alpha channel arrays.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if normalize:
        img = img.astype(np.float64) / 255

    alpha = img[:, :, -1]
    img = img[:, :, :-1]

    return img, alpha


def imshow(img: np.ndarray, title: str = None, flag: int = cv2.COLOR_BGR2RGB):
    """
    Display an image.

    Args:
        img (np.ndarray): Image array.
        title (str, optional): Plot title. Defaults to None.
        flag (int, optional): cv2 color conversion flag. Defaults to cv2.COLOR_BGR2RGB.
    """
    plt.figure()
    if flag is not None:
        img = cv2.cvtColor(img, flag)
    plt.imshow(img)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()


def imwrite(path: Path, img: np.ndarray, flag: int = None):
    """
    Write an image to file.

    Args:
        path (Path): Image path.
        img (np.ndarray): Image array.
        flag (int, optional): cv2 color conversion flag. Defaults to None.
    """
    assert type(img) == np.ndarray
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    if flag is not None:
        img = cv2.cvtColor(img, flag)
    cv2.imwrite(str(path), img)


def binarize(
    img: np.ndarray, thresh: int = 60, fill_holes: bool = False, debug: bool = False
) -> np.ndarray:
    """
    Binarize an image.

    Args:
        img (np.ndarray): Input image.
        thresh (int, optional): Intensity value for the binary theshold. Defaults to 60.
        fill_holes (bool, optional): Whether or not to inpaint holes in the image. Defaults to False.
        debug (bool, optional): Debug flag to display the image. Defaults to False.

    Returns:
        np.ndarray: A binarized version of the input image.
    """
    _, binary_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    if fill_holes:
        binary_img = (ndimage.binary_fill_holes(binary_img) * 255).astype(np.uint8)
    if debug:
        imshow(binary_img)
    return binary_img


def get_radii(debug: bool = False) -> Dict[int, Dict[str, Any]]:
    """
    Find the radius of each coin in `coins_1.png`.

    Args:
        debug (bool, optional): Debug flag to print intermediate information. Defaults to False.

    Returns:
        Dict[int, Dict[str, Any]]: A dictionary containing the information for the coins.
    """
    # Read the image as a grayscale image
    img = imread(get_data_path("coins_1.png"), cv2.IMREAD_GRAYSCALE)

    # Binarize the grayscale image
    binary_img = binarize(img, thresh=60, debug=debug)

    # Segment binary image into connected regions
    num_labels, labeled_img = cv2.connectedComponents(binary_img)

    # Set background label as 0
    labeled_img[binary_img == 0] = 0

    # Find the radius of each coin
    if debug:
        print("Number of coins: ", num_labels - 1)
    coins = []
    for i in range(num_labels - 1):
        # Retrieve the connected component (each coin) as a binary image
        coin_i_img = labeled_img.copy()
        coin_i_img = np.where(coin_i_img == i + 1, 1, 0)

        # Find the radius of the coin and its center location
        bw_dist = ndimage.distance_transform_edt(coin_i_img)
        r = bw_dist.max()
        max_idx = np.where(bw_dist == bw_dist.max())
        x_c = max_idx[1][0]
        y_c = max_idx[0][0]

        if debug:
            print("Coin ", i + 1)
            print(f"Center (x,y): ({x_c}, {y_c})")
            print(f"Radius: {r}")

        coins.append({"center_r": int(y_c), "center_c": int(x_c), "radius": int(r)})

        # Visualize the coin
        if debug:
            ig, ax = plt.subplots()
            ax.imshow(img)
            circle = patches.Circle((x_c, y_c), r, color="r", alpha=0.5)
            ax.add_patch(circle)
            plt.show()
            plt.close()

    # Post process coins information
    coin_names = ["dime", "nickel", "quarter"]
    coin_values = [10, 5, 25]
    coin_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    coins_info = {}
    coins = sorted(coins, key=lambda k: k["radius"])
    for i, c in enumerate(coins):
        coins_info[c["radius"]] = {
            "name": coin_names[i],
            "value": coin_values[i],
            "color": coin_colors[i],
        }

    if debug:
        print(coins_info)

    return coins_info


def annotate_coins(
    img: np.ndarray,
    detected_coins: List[Tuple[int, int, int, int]],
    coins_info: Dict[int, Dict[str, Any]],
    radius_range: int = 5,
) -> np.ndarray:
    """
    Annotate the coins in an image.

    Args:
        img (np.ndarray): The input image.
        detected_coins (List[Tuple[int, int, int, int]]): Information of the detected
            coins -- a list of (x, y, r, votes).
        coins_info (Dict[int, Dict[str, Any]]): A dictionary containing the information for the coins.
        radius_range (int, optional): The range of radius to search for. Defaults to 5.

    Returns:
        np.ndarray: An annotated image highlighting the coins.
    """
    circle_detected_img = img.copy()
    for x, y, r, _ in detected_coins:
        for radius, info in coins_info.items():
            if abs(radius - r) <= radius_range:
                circle_detected_img = cv2.circle(
                    circle_detected_img, (x, y), r, info["color"], 5
                )
    return circle_detected_img
