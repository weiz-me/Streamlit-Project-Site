from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils_hw3 as utils
import math
from skimage import morphology


# --------------------------------------------------------------------------
# Academic Honesty Policy
# --------------------------------------------------------------------------

# The below credentials are equivalent to signing the academic honesty policy
STUDENT_NAME = "Wei Zhang"  # TODO: Fill in your name
STUDENT_UNI = "wz2580"  # TODO: Fill in your UNI


def sign_academic_honesty_policy():
    assert (
        STUDENT_NAME is not None and STUDENT_UNI is not None
    ), "Please fill in your STUDENT_NAME and STUDENT_UNI at the top of solutions_UNI.py"
    print(
        f"I, {STUDENT_NAME} ({STUDENT_UNI}), certify that I have read and agree to the"
        " Code of Academic Integry."
    )


# --------------------------------------------------------------------------
# Walkthrough 1: Image processing
# --------------------------------------------------------------------------


def walkthrough1a():
    """
    Image processing: convolution, Gaussian smoothing

    Image credit: http://commons.wikimedia.org/wiki/File:Beautiful-pink-flower_-_West_Virginia_-_ForestWander.jpg
    """
    # Load the image
    img = utils.imread(utils.get_data_path("flower.png"))
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Define the sigma values
    sigma = [6, 12, 24]
    # Define the kernel sizes
    k = [int(np.ceil(2 * np.pi * s)) // 2 * 2 + 1 for s in sigma]
    # Create a figure with 2 rows and 2 columns of subplots
    (fig, axs) = plt.subplots(2, 2)
    # Display the original image
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 0].axis("off")
    axs[0, 0].set_title("Original")
    # Loop through the sigma values and apply Gaussian blur
    for i in range(len(sigma)):
        # Generate a Gaussian kernel
        h = cv2.GaussianBlur(img_gray, (k[i], k[i]), sigma[i])
        # Display the blurred image
        axs[(i + 1) // 2, (i + 1) % 2].imshow(h, cmap="gray")
        axs[(i + 1) // 2, (i + 1) % 2].axis("off")
        axs[(i + 1) // 2, (i + 1) % 2].set_title(f"sigma = {sigma[i]}")
    # Set the plot layout and save the figure
    fig.tight_layout()
    plt.savefig(utils.get_result_path("w1a_blur-flowers.png"))
    # Show result plot
    plt.show()


def walkthrough1b():
    """
    Edge detection

    Image credit: CAVE Lab
    """
    # Load the image
    img = utils.imread(utils.get_data_path("hello.png"))
    # Display the color image
    (fig, axs) = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 0].axis("off")
    axs[0, 0].set_title("Color Image")
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    axs[0, 1].imshow(gray_img, cmap="gray")
    axs[0, 1].axis("off")
    axs[0, 1].set_title("Grayscale Image")
    # Sobel edge detection
    thresh = 0.05 * np.max(gray_img)
    edge_img = cv2.Sobel(gray_img, cv2.CV_64F, 1, 1, ksize=3)
    edge_img[edge_img < thresh] = 0
    edge_img[edge_img >= thresh] = 255
    edge_img = edge_img.astype(np.uint8)
    axs[1, 0].imshow(edge_img, cmap="gray")
    axs[1, 0].axis("off")
    axs[1, 0].set_title("Sobel Edge Detection")
    # Canny edge detection
    thresh1 = 0.05 * np.max(gray_img)
    thresh2 = 0.1 * np.max(gray_img)
    edge_img = cv2.Canny(gray_img, thresh1, thresh2)
    axs[1, 1].imshow(edge_img, cmap="gray")
    axs[1, 1].axis("off")
    axs[1, 1].set_title("Canny Edge Detection")
    # Save the resulting image
    plt.savefig(utils.get_result_path("w1b_hello-edges.png"))
    # Show result plot
    plt.show()


# --------------------------------------------------------------------------
# Challenge 1: Line Detection
# --------------------------------------------------------------------------


def find_edge_pixels(img: np.ndarray) -> np.ndarray:
    """
    Use the Canny edge detection function from OpenCV

    Args:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The edge image returned by Canny.
    """

    # Canny edge detection
    gray_img = img
    thresh1 = 0.49 * np.max(gray_img)
    thresh2 = 0.50 * np.max(gray_img)
    edge_img = cv2.Canny(gray_img, thresh1, thresh2)
    # plt.imshow(edge_img, cmap="gray")
    return edge_img
    # raise NotImplementedError()


def generate_hough_accumulator(edge_img: np.ndarray) -> np.ndarray:
    """
    Generate the Hough accumulator for the edge image.

    Args:
        edge_img (np.ndarray): The edge image resultant from Canny edge detection.

    Returns:
        np.ndarray: The Hough accumulator (scaled from 0 to 255).
    """
    """
    Comment: I look through the neighbor of hough_accumulator, 
    and set it to zero if it's not the maximum within the neightbor 
    """
    # go through each pixel
    m = len(edge_img)
    n = len(edge_img[0])

    # print(f"m = {m}, n = {n}")
    # print(edge_img.shape)
    rho_size = int((m**2+n**2)**(0.5))
    theta_size = 314*2
    Hough = np.zeros((rho_size*2,theta_size),dtype=int)

    # print(Hough)
    for y in range(m):
        for x in range(n):
            if edge_img[y][x] == 0:
                continue
            # print(f"{y}, {x}: {edge_img[y][x]}")

            # convert to sigmoid function
            for theta in range(theta_size):
                rho = y*math.cos(theta/100) - x * math.sin(theta/100)
                Hough[int(rho)+rho_size][theta]+=1    

                # Hough[int(rho)+rho_size][theta][0]+=1    
                # Hough[int(rho)+rho_size][theta][1]+=1    
                # Hough[int(rho)+rho_size][theta][2]+=1    

            # plt.imshow(Hough, cmap="gray")
            # plt.show()

    # plt.imshow(Hough, cmap="gray")
    # # print(Hough)
            
    # max_val = np.max(Hough)
    # Hough = 255 * (Hough) / (max_val )

    thold = 10
    Hough[Hough<=thold]=0
    # make a lookup table for max
    l_table = np.zeros(Hough.shape)
    index_value = []
    neighbor = 30
    # history = set()
    for y in range(rho_size*2):
        for x in range(theta_size):
            if Hough[y][x] == 0:
                continue
            index_value.append([Hough[y][x],y,x])
    
    index_value=sorted(index_value)
    while index_value:
        h,y,x = index_value.pop(0)
        # print(f"{h}, {y}, {x}")
        l1 = max(0,y-neighbor//4)
        l3 = min(rho_size*2,y+neighbor//4+1)
        l2 = max(0,x-neighbor)
        l4 = min(theta_size, x+neighbor+1)
        l_table[l1:l3,l2:l4] = h


        # for i in range(l1,l3):
        #     for j in range(l2,l4):
        #         # if (i,j) in history:
        #         #     continue
        #         l_table[i][j] = h
        #         history.add((i,j))
    
    # print(l_table)

    




    for y in range(rho_size*2):
        l1 = max(0,y-neighbor//4)
        l3 = min(rho_size*2,y+neighbor//4+1)

        for x in range(theta_size):
            cur_val = Hough[y][x]
            if cur_val == 0:
                continue
            # look at neighbor
            l2 = max(0,x-neighbor)
            l4 = min(theta_size, x+neighbor+1)
            # print(f"{l1},{l3},{l2},{l4}")
            # neigh = Hough[l1:l3]
            # neigh = neigh[:,l2:l4]
            # # print(neigh.shape)
            # n_max = np.max(neigh)
            if cur_val < l_table[y][x]:
                Hough[y][x] = 0
            
            

    img_float32 = np.uint8(Hough)


    return img_float32






    # raise NotImplementedError()


def line_finder(orig_img: np.ndarray, hough_img: np.ndarray,threshold = 110) -> np.ndarray:
    """
    Find the lines in the original image using the Hough accumulator.

    Args:
        orig_img (np.ndarray): The original image to find lines in.
        hough_img (np.ndarray): The Hough accumulator.

    Returns:
        np.ndarray: The original image overlaid with lines.
    """


    h_max= hough_img.max()
    m = len(orig_img)
    n = len(orig_img[0])
    rho = len(hough_img)
    theta = len(hough_img[0])
    
    lines = []
    for r in range(rho):
        for t in range(theta):
            if hough_img[r][t] <threshold:
                continue
            # print(f"r {r}, t {t} : {hough_img[r][t]}")

            x1=0
            y1= (r-rho//2 + x1*math.sin(t/100))/(math.cos(t/100))

            while y1<0 or y1>=m:
                x1+=1
                y1= (r-rho//2 + x1*math.sin(t/100))/(math.cos(t/100))
                # print(f"x1 {x1}, y1 {y1}")

                if x1>=n:
                    break
            # print(f"x1 {x1}, y1 {y1}")

            if x1>=n:
                continue

            x2=n
            y2= (r-rho/2  + x2*math.sin(t/100))/math.cos(t/100)

            while y2<0 or y2>=m:
                x2-=1
                y2= (r-rho/2  + x2*math.sin(t/100))/math.cos(t/100)
                # print(f"x2 {x2}, y2 {y2}")

                if x2<0:
                    break

            if x2<0:
                continue
            slope = 0
            if (x2-x1):
                slope = (y2-y1)/(x2-x1)
            b = (y1-m*x1)
            skip = False
            # for l in lines:
            #     if abs(l[1]-slope)<0.5:
            #         if abs(l[3] -y1)<15:
            #             skip = True
            #             break
            #         if abs(l[4] -y2)<15:
            #             skip = True
            #             break
            #     if abs(slope) >5:
            #         if abs(x1-l[2]) < 15:
            #             skip = True
            #             break
            #         if abs(x2-l[5]) < 15:
            #             skip = True
            #             break

            
            lines.append((b,slope,x1,y1,y2,x2))
            # if skip:
            #     continue
            start_point = (int(x1), int(y1))
            end_point = (int(x2), int(y2))


            cv2.line(orig_img, start_point,end_point,(0,255,0) , 2)

            # print(f"start_point {start_point}, end {start_point}")

    # lines_np = np.array(lines)
    # print(lines_np.shape)

    # # cv2.line(orig_img, start_point,end_point,(0,255,0) , 2)
    # slope = lines_np[:,5]
    # b = lines_np[:,6]
    # h = lines_np[:,4]

    # indices = np.arange(len(x1))
    # # print(indices)

    # for i, line in enumerate(lines_np):

        
    #     if i not in indices:
    #         continue
        
    #     slope = lines_np[:,5]
    #     b = lines_np[indices,6]
    #     h = lines_np[:,4]



    #     x1_diff = x1-line[0]
    #     y1_diff = y1-line[1]
    #     x2_diff = x2-line[2]
    #     y2_diff = y2-line[3]

    #     diff = np.absolute(x1_diff)+np.absolute(y1_diff)+np.absolute(x2_diff)+np.absolute(y2_diff)
    #     print(diff.shape)


    #     line_diff = 0
    return orig_img
    # print(h_max)
    # raise NotImplementedError()


# --------------------------------------------------------------------------
# Challenge 2: Coin Counting
# --------------------------------------------------------------------------


def generate_coin_hough_accumulator_2b(
    orig_img: np.ndarray, coins_info: Dict[int, Dict[str, Any]], radius_range: int = 0
) -> Dict[Tuple[int, int, int], int]:
    """
    Generate the Circle Hough accumulators for coin detection

    Args:
        orig_img (np.ndarray): The original image to find coins in.
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        radius_range (int, optional): The range of radius to search for. Defaults to 0.

    Returns:
        Dict[Tuple[int, int, int], int]: The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.
    """
    # raise NotImplementedError()
    img = orig_img
    # print(coins_info)
    binary_img = utils.binarize(img, thresh=40)
    # utils.imshow(binary_img)

    gray_img = binary_img
    thresh1 = 0.2 * np.max(gray_img)
    thresh2 = 0.21 * np.max(gray_img)
    edge_img = cv2.Canny(gray_img, thresh1, thresh2)
    # utils.imshow(edge_img)

    yn = len(img)
    xn = len(img[0])
    # rmap = {}
    output={}

    for i,r in enumerate(coins_info.keys()):
        print(f"r {r}")
        # rmap[i]=r
        b_size = (r**2 + xn**2)**(0.5)+yn
        a_size =  (r**2 + yn**2)**(0.5) + xn

        a_size = int(a_size)
        for y in range(yn):
            for x in range(xn):
                if edge_img[y][x] == 0:
                    continue

                for a in range(a_size):
                    temp = r**2 - (a-x)**2
                    if temp <0:
                        continue

                    elif temp == 0:
                        if (a,y,r) not in output.keys():
                            output[(a,y,r)] = 0
                        output[(a,y,r)] +=1
                    else:
                        diff = int(temp**(0.5))
                        if (a,y+diff,r) not in output.keys():
                            output[(a,y+diff,r)] = 0
                        output[(a,y+diff,r)] +=1

                        if (a,y-diff,r) not in output.keys():
                            output[(a,y-diff,r)] = 0
                        output[(a,y-diff,r)] +=1
        # print(output)
    return output
        
                


def coin_finder_2b(
    coin_hough: Dict[Tuple[int, int, int], int]
) -> List[Tuple[int, int, int, int]]:
    """
    Find the coins using the Circle Hough accumulator.

    Args:
        coin_hough (Dict[Tuple[int, int, int], int]): The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.

    Returns:
        List[Tuple[int, int, int, int]]: Information of the detected coins -- a list of (x, y, r, votes).
    """
    output = []
    threhold = 60
    rt = 200

    for c in coin_hough.keys():
        if coin_hough[c] >= threhold:
            output.append((c[0],c[1],c[2],coin_hough[c]))
    # print(output)
    output = np.array(output)
    sorted_indices = output[:, -1].argsort()
    sorted_indices_reversed = sorted_indices[::-1]

    sorted_array = output[sorted_indices_reversed]
    detect_coin = []
    history = set()

    for row in sorted_array:
        skip = False
        for l in history:
            diff = abs(row[0]-l[0]) + abs(row[1]-l[1])
            if diff < (rt):
                skip = True
                break
                
        if skip:
            continue

        history.add((row[0],row[1]))
        detect_coin.append(row)

        # if row[3]<threhold:
        #     break
    
    return detect_coin

    # raise NotImplementedError()



def get_total_value_2b(
    coins_info: Dict[int, Dict[str, Any]],
    detected_coins: List[Tuple[int, int, int, int]],
    radius_range: int = 10,
) -> float:
    """
    Find the total value of all the detected coins

    Args:
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        detected_coins (List[Tuple[int, int, int, int]]): Information of the detected coins -- a
            list of (x, y, r, votes).
        radius_range (int, optional): The range of radius to search for. Defaults to 10.

    Returns:
        float: The total value in USD.
    """
    # raise NotImplementedError()
    # print(coins_info)

    output = 0
    for coin in detected_coins:
        output += coins_info[coin[2]]['value']/100
    return output
        


def generate_coin_hough_accumulator_2c(
    orig_img: np.ndarray, coins_info: Dict[int, Dict[str, Any]], radius_range: int = 0
) -> Dict[Tuple[int, int, int], int]:
    """
    Generate the Circle Hough accumulators for coin detection

    Args:
        orig_img (np.ndarray): The original image to find coins in.
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        radius_range (int, optional): The range of radius to search for. Defaults to 0.

    Returns:
        Dict[Tuple[int, int, int], int]: The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.
    """
    img = orig_img
    # print(coins_info)
    binary_img = utils.binarize(img, thresh=40)
    # utils.imshow(binary_img)

    gray_img = binary_img
    # print(gray_img.shape)
    # print(gray_img)
    thresh1 = 0.01 * np.max(gray_img)
    thresh2 = 0.02 * np.max(gray_img)
    edge_img = cv2.Canny(gray_img, thresh1, thresh2)
    # utils.imshow(edge_img)


    yn = len(img)
    xn = len(img[0])
    # rmap = {}
    output={}

    for i,r in enumerate(coins_info.keys()):
        print(f"r {r}")
        # rmap[i]=r
        b_size = (r**2 + xn**2)**(0.5)+yn
        a_size =  (r**2 + yn**2)**(0.5) + xn

        a_size = int(a_size)
        for y in range(yn):
            for x in range(xn):
                if edge_img[y][x] == 0:
                    continue

                for a in range(a_size):
                    temp = r**2 - (a-x)**2
                    if temp <0:
                        continue

                    elif temp == 0:
                        if (a,y,r) not in output.keys():
                            output[(a,y,r)] = 0
                        output[(a,y,r)] +=1
                    else:
                        diff = int(temp**(0.5))
                        if (a,y+diff,r) not in output.keys():
                            output[(a,y+diff,r)] = 0
                        output[(a,y+diff,r)] +=1

                        if (a,y-diff,r) not in output.keys():
                            output[(a,y-diff,r)] = 0
                        output[(a,y-diff,r)] +=1
        # print(output)
    return output
        
                

def coin_finder_2c(
    coin_hough: Dict[Tuple[int, int, int], int]
) -> List[Tuple[int, int, int, int]]:
    """
    Find the coins using the Circle Hough accumulator.

    Args:
        coin_hough (Dict[Tuple[int, int, int], int]): The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.

    Returns:
        List[Tuple[int, int, int, int]]: Information of the detected coins -- a list of (x, y, r, votes).
    """
    # raise NotImplementedError()
    output = []
    threhold = 60
    rt = 60

    for c in coin_hough.keys():
        if coin_hough[c] >= threhold:
            output.append((c[0],c[1],c[2],coin_hough[c]))
    # print(output)
    output = np.array(output)
    sorted_indices = output[:, -1].argsort()
    sorted_indices_reversed = sorted_indices[::-1]

    sorted_array = output[sorted_indices_reversed]
    detect_coin = []
    history = set()

    for row in sorted_array:
        skip = False
        for l in history:
            diff = abs(row[0]-l[0]) + abs(row[1]-l[1])
            if diff < (rt):
                skip = True
                break
                
        if skip:
            continue

        history.add((row[0],row[1]))
        detect_coin.append(row)

        # if row[3]<threhold:
        #     break
    
    return detect_coin



def get_total_value_2c(
    coins_info: Dict[int, Dict[str, Any]],
    detected_coins: List[Tuple[int, int, int, int]],
    radius_range: int = 10,
) -> float:
    """
    Find the total value of all the detected coins

    Args:
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        detected_coins (List[Tuple[int, int, int, int]]): Information of the detected coins -- a
            list of (x, y, r, votes).
        radius_range (int, optional): The range of radius to search for. Defaults to 10.

    Returns:
        float: The total value in USD.
    """
    
    output = 0
    for coin in detected_coins:
        output += coins_info[coin[2]]['value']/100
    return output
        

    # raise NotImplementedError()


def generate_coin_hough_accumulator_2d(
    orig_img: np.ndarray, coins_info: Dict[int, Dict[str, Any]], radius_range: int = 0
) -> Dict[Tuple[int, int, int], int]:
    """
    Generate the Circle Hough accumulators for coin detection

    Args:
        orig_img (np.ndarray): The original image to find coins in.
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        radius_range (int, optional): The range of radius to search for. Defaults to 0.

    Returns:
        Dict[Tuple[int, int, int], int]: The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.
    """
    img = orig_img
    threshold = 40
    bw_img = np.zeros_like(img)
    bw_img[img > threshold] = 255
    binary_img = img > threshold
    # plt.imshow(bw_img, cmap='gray')
    # plt.show()
    k = 1

    processed_img = morphology.binary_dilation(binary_img[:, :, 1],
                                               morphology.disk(k))
    processed_img = morphology.binary_erosion(processed_img,
                                              morphology.disk(k))
    # plt.imshow(processed_img, cmap='gray')
    # plt.show()
    k = 80

    processed_img = morphology.binary_erosion(processed_img, morphology.disk(k))
    
    processed_img = morphology.binary_dilation(processed_img, morphology.disk(k))
    plt.imshow(processed_img, cmap='gray')
    plt.show()

    gray_img = processed_img
    color_image = np.zeros((gray_img.shape[0], gray_img.shape[1], 3), dtype=np.uint8)

    # Copy the grayscale data to all three channels
    color_image[:, :, 0] = gray_img
    color_image[:, :, 1] = gray_img
    color_image[:, :, 2] = gray_img
    # print(gray_img.shape)

    thresh1 = 0.40 * np.max(color_image)
    thresh2 = 0.41 * np.max(color_image)
    edge_img = cv2.Canny(color_image, thresh1, thresh2)
    utils.imshow(edge_img)

    yn = len(img)
    xn = len(img[0])
    # rmap = {}
    output={}

    for i,r in enumerate(coins_info.keys()):
        print(f"r {r}")
        # rmap[i]=r
        b_size = (r**2 + xn**2)**(0.5)+yn
        a_size =  (r**2 + yn**2)**(0.5) + xn

        a_size = int(a_size)
        for y in range(yn):
            for x in range(xn):
                if edge_img[y][x] == 0:
                    continue

                for a in range(a_size):
                    temp = r**2 - (a-x)**2
                    if temp <0:
                        continue

                    elif temp == 0:
                        if (a,y,r) not in output.keys():
                            output[(a,y,r)] = 0
                        output[(a,y,r)] +=1
                    else:
                        diff = int(temp**(0.5))
                        if (a,y+diff,r) not in output.keys():
                            output[(a,y+diff,r)] = 0
                        output[(a,y+diff,r)] +=1

                        if (a,y-diff,r) not in output.keys():
                            output[(a,y-diff,r)] = 0
                        output[(a,y-diff,r)] +=1
        # print(output)
    return output

def coin_finder_2d(
    coin_hough: Dict[Tuple[int, int, int], int]
) -> List[Tuple[int, int, int, int]]:
    """
    Find the coins using the Circle Hough accumulator.

    Args:
        coin_hough (Dict[Tuple[int, int, int], int]): The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.

    Returns:
        List[Tuple[int, int, int, int]]: Information of the detected coins -- a list of (x, y, r, votes).
    """
    # raise NotImplementedError()
    output = []
    threhold = 60
    rt = 60

    for c in coin_hough.keys():
        if coin_hough[c] >= threhold:
            output.append((c[0],c[1],c[2],coin_hough[c]))
    # print(output)
    output = np.array(output)
    sorted_indices = output[:, -1].argsort()
    sorted_indices_reversed = sorted_indices[::-1]

    sorted_array = output[sorted_indices_reversed]
    detect_coin = []
    history = set()

    for row in sorted_array:
        skip = False
        for l in history:
            diff = abs(row[0]-l[0]) + abs(row[1]-l[1])
            if diff < (rt):
                skip = True
                break
                
        if skip:
            continue

        history.add((row[0],row[1]))
        detect_coin.append(row)

        # if row[3]<threhold:
        #     break
    
    return detect_coin


def get_total_value_2d(
    coins_info: Dict[int, Dict[str, Any]],
    detected_coins: List[Tuple[int, int, int, int]],
    radius_range: int = 10,
) -> float:
    """
    Find the total value of all the detected coins

    Args:
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        detected_coins (List[Tuple[int, int, int, int]]): Information of the detected coins -- a
            list of (x, y, r, votes).
        radius_range (int, optional): The range of radius to search for. Defaults to 10.

    Returns:
        float: The total value in USD.
    """

    output = 0
    for coin in detected_coins:
        output += coins_info[coin[2]]['value']/100
    return output
        

    # raise NotImplementedError()