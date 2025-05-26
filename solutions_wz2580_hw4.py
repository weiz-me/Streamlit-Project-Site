from pathlib import Path
from typing import Tuple, List

import numpy as np
from scipy.ndimage import distance_transform_edt
import cv2

import utils_hw4 as utils
import matplotlib.pyplot as plt


# Tunable parameters

# RANSAC parameters for challenge1a
CHALLENGE_1C_RANSAC_N = 500 # Number of iterations
CHALLENGE_1C_RANSAC_EPS = 5 # Maximum reprojection error


# --------------------------------------------------------------------------
# Academic Honesty Policy
# --------------------------------------------------------------------------
# The below credentials are equivalent to signing the academic honsety policy
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


#--------------------------------------------------------------------------
# Challenge 1: Image Mosaicking App
#--------------------------------------------------------------------------
def compute_homography(src_pts: np.ndarray, dest_pts: np.ndarray) -> np.ndarray:
    """
    Compute the homography matrix relating the given points.
    Hint: use np.linalg.eig to compute the eigenvalues of a matrix.

    Args:
        src_pts (np.ndarray): Nx2 matrix of source points
        dest_pts (np.ndarray): Nx2 matrix of destination points

    Returns:
        np.ndarray: 3x3 homography matrix
    """
    assert src_pts.shape[0] == dest_pts.shape[0]
    assert src_pts.shape[1] == 2 and dest_pts.shape[1] == 2
    N = src_pts.shape[0]

    # print(src_pts)
    # print(dest_pts)
    numofpts = src_pts.shape[0]
    A = np.zeros((2*numofpts,9))
    # print(A)

    for i in range(numofpts):
        xs = src_pts[i][0]
        ys = src_pts[i][1]
        xd = dest_pts[i][0]
        yd = dest_pts[i][1]
        temp_np = np.array([xs,ys,1])

        A[2*i,:3] = temp_np
        A[2*i+1,3:6] = temp_np
        A[2*i,6:] = temp_np * (-xd)
        A[2*i+1,6:] = temp_np * (-yd)
    
    # print(A)

    eigenvalues, eigenvectors = np.linalg.eig(A.T @ A)
    min_index = np.argmin(eigenvalues)
    min_eigenvector = eigenvectors[:,min_index]
    # print(eigenvalues)
    # print(eigenvectors)
    # print(min_index)
    # print(min_eigenvector)

    H = min_eigenvector.reshape(3,3)
    # print(H)
    return H

    # raise NotImplementedError()


def apply_homography(H: np.ndarray, test_pts: np.ndarray) -> np.ndarray:
    """
    Apply the homography to the given test points

    Args:
        H (np.ndarray): 3x3 homography matrix
        test_pts (np.ndarray): Nx2 test points

    Returns:
        np.ndarray: Nx2 points after applying the homography
    """
    assert test_pts.shape[1] == 2

    # print(H)
    # print(test_pts)
    numofpts = test_pts.shape[0]
    src_pts = np.ones((3,numofpts))
    src_pts[0:2,:] = test_pts.T
    # print(src_pts)

    des_pts = H @ src_pts
    bot_row = des_pts[-1,:]
    scaled_des_pts = des_pts / bot_row

    # print(des_pts)
    # print(scaled_des_pts)

    ret_pts = scaled_des_pts[:2,:].T
    # print(ret_pts)

    return ret_pts

    # raise NotImplementedError()


def backward_warp_img(
    src_img: np.ndarray, H_inv: np.ndarray, 
    dest_canvas_width_height: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a homography to the image using backward warping.

    Use cv2.remap to linearly interpolate the warped points.
    The function call should follow this form:
        img_warp = cv2.remap(img, map_x.astype(np.float32), 
            map_y.astype(np.float32), cv2.INTER_LINEAR, borderValue=np.nan).
    This casts map_x and map_y to 32-bit floats, chooses linear interpolation,
    and sets pixels outside the original image to NaN (not-a-number).
        
    Also, since we are working with color images, you should process each
    color channel separately.

    Args:
        src_img (np.ndarray): Nx2 source points
        H_inv (np.ndarray): 3x3 inverse of the src -> dest homography
        dest_canvas_width_height (Tuple[int, int]): size of the destination image

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
        binary mask where the destination image is filled in, final image
    """

    dest_width,dest_height = dest_canvas_width_height

    dest_pts = np.array([[x,y,1] for y in range(dest_height) for x in range(dest_width)])

    dest_inputs = dest_pts.T

    src_pts_output = H_inv @ dest_inputs
    scaled_src_pts_output = src_pts_output / src_pts_output[-1,:]
    src_pts = scaled_src_pts_output[:2,:].T

    map_x = src_pts[:,0].astype(np.float32)
    map_x = map_x.reshape(dest_height,dest_width)
    map_y = src_pts[:,1].astype(np.float32)
    map_y = map_y.reshape(dest_height,dest_width)

    img_warp_r = cv2.remap(src_img[:,:,0], map_x, map_y, cv2.INTER_LINEAR, borderValue=np.nan)
    img_warp_g = cv2.remap(src_img[:,:,1], map_x, map_y, cv2.INTER_LINEAR, borderValue=np.nan)
    img_warp_b = cv2.remap(src_img[:,:,2], map_x, map_y, cv2.INTER_LINEAR, borderValue=np.nan)

    img_warp = np.dstack((img_warp_r,img_warp_g,img_warp_b))
    binary_mask = ~np.isnan(img_warp)

    return binary_mask,img_warp
    # raise NotImplementedError()


def warp_img_onto(src_img: np.ndarray, dest_img: np.ndarray, 
                  src_pts: np.ndarray, dest_pts: np.ndarray) -> np.ndarray:
    """
    Warp the source image on the destination image.
    Return the resulting image.
    
    Step 1: estimate the homography mapping src_pts to dest_pts
    Step 2: warp src_img onto dest_img using backward_warp_img(..)
    Step 3: overlay the warped src_img onto dest_img
    Step 4: return the resulting image

    Args:
        src_img (np.ndarray): source image
        dest_img (np.ndarray): destination image
        src_pts (np.ndarray): Nx2 source points
        dest_pts (np.ndarray): Nx2 destination points

    Returns:
        np.ndarray: resulting image with the source image warped on the 
        destination image
    """
    H = compute_homography(src_pts, dest_pts)
    H_inv = np.linalg.inv(H)
    # print(H_inv.shape)
    
    dest_height = dest_img.shape[0]
    dest_width = dest_img.shape[1]
    dest_canvas_width_height = dest_width,dest_height

    mask, img_warp = backward_warp_img(src_img,H_inv, dest_canvas_width_height)
    # print(mask)
    # print(dest_img[mask])
    # print(img_warp[mask])
    # print(np.max(dest_img[~mask]))
    # print(np.max(img_warp[mask]))
    # utils.imshow(dest_img, flag=None)
    # utils.imshow(img_warp, flag=None)


    # output_img = np.zeros_like(dest_img)
    # output_img[mask]=img_warp[mask]
    # output_img[~mask][0]=dest_img[~mask][0]
    # output_img[~mask][1]=dest_img[~mask][1]
    # output_img[~mask][2]=dest_img[~mask][2]

    dest_img[mask]=img_warp[mask]
    return dest_img
    # mask0 = np.all(dest_img == [0,0,0],axis =-1)
    # utils.imshow(mask0)
    # return blend_image_pair(dest_img,mask0,img_warp,mask,'blend')

    # raise NotImplementedError()


def run_RANSAC(src_pts: np.ndarray, dest_pts: np.ndarray, ransac_n: int, 
               ransac_eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run RANSAC on the given point correspondences to compute a homography 
    from source to destination points.

    Args:
        src_pts (np.ndarray): Nx2 source points
        dest_pts (np.ndarray): Nx2 destination points
        ransac_n (int): number of RANSAC iterations
        ransac_eps (float): maximum 2D reprojection error for inliers

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of the inlier indices and the 
        estimated homography matrix.
    """
    assert src_pts.shape[0] == dest_pts.shape[0]
    assert src_pts.shape[1] == 2 and dest_pts.shape[1] == 2

    # print(src_pts)
    # print(dest_pts)

    inlier_indices= None
    estimated_H = None
    inlier_num = 0
    best_ind = None

    src_pts_input = np.ones((3,src_pts.shape[0]))
    src_pts_input[:2,:]=src_pts.T
    # print(src_pts_input)

    for _ in range(ransac_n):
        random_indices = np.random.choice(src_pts.shape[0],4)
        curr_src_pts = src_pts[random_indices]
        curr_dest_pts = dest_pts[random_indices]
        # print(curr_src_pts)
        # print(curr_dest_pts)

        curr_H = compute_homography(curr_src_pts,curr_dest_pts)
        dest_pts_output = curr_H @ src_pts_input
        scaled_dest_pts_output = dest_pts_output / dest_pts_output[-1,:]
        estimated_dest_pts = scaled_dest_pts_output[:2,:].T
        # print(estimated_dest_pts)

        pts_diff = dest_pts-estimated_dest_pts
        pts_dist = np.linalg.norm(pts_diff,axis=1)

        # print(pts_diff)
        # print(pts_dist)

        curr_inlier_indices = np.where(pts_dist < ransac_eps)[0]
        curr_inlier_num = curr_inlier_indices.shape[0]
        # print(curr_inlier_indices)
        # print(curr_inlier_num)

        if curr_inlier_num > inlier_num:
            inlier_num = curr_inlier_num
            inlier_indices = curr_inlier_indices
            estimated_H = curr_H
            best_ind = random_indices
        

    
    return best_ind,estimated_H
    # raise NotImplementedError()


def blend_image_pair(img1: np.ndarray, mask1: np.ndarray, img2: np.ndarray, 
                     mask2: np.ndarray, blending_mode: str) -> np.ndarray:
    """
    Blend two images together using the "overlay" or "blend" mode.
    
    In the "overlay" mode, image 2 is overlaid on top of image 1 wherever
    mask2 is non-zero.
    In "blend" mode, the blended image is a weighted combination of the two
    images, where each pixel is weighted based on its distance to the edge.

    Args:
        img1 (np.ndarray): First image
        mask1 (np.ndarray): Mask where the first image is non-zero
        img2 (np.ndarray): Second image
        mask2 (np.ndarray): Mask where the second image is non-zero
        blending_mode (str): "overlay" or "blend"

    Returns:
        np.ndarray: blended image.
    """
    assert blending_mode in ["overlay", "blend"]

    # print(mask1.shape)
    # print(mask1)
    # print(mask2.shape)
    # print(mask2)
    bmask1 = mask1.astype(np.bool_)
    # print(bmask1)

    bmask2 = mask2.astype(np.bool_)
    
    if blending_mode == "overlay":
        result = np.copy(img1)
        result[bmask2] = img2[bmask2]

    elif blending_mode == "blend":

        dist_mask1 = distance_transform_edt(bmask1)
        dist_mask2 = distance_transform_edt(bmask2)

        # Normalize the distance maps to [0, 1]
        dist_mask1 /= np.max(dist_mask1)
        dist_mask2 /= np.max(dist_mask2)

        dist_mask1 = np.dstack((dist_mask1,dist_mask1,dist_mask1))
        dist_mask2 = np.dstack((dist_mask2,dist_mask2,dist_mask2))


        # Combine the two images based on the distance maps
        img1 = img1.astype(float)
        img2 = img2.astype(float)
        overlap = np.logical_and(bmask1,bmask2)
        result = np.copy(img1)
        result[bmask2] = img2[bmask2]
        result[overlap] = (((img1 * dist_mask1) + (img2 * dist_mask2))/(dist_mask1+dist_mask2))[overlap]
        result[np.logical_and(~bmask1,bmask2)]= img2[np.logical_and(~bmask1,bmask2)]
        result[np.logical_and(bmask1,~bmask2)]= img1[np.logical_and(bmask1,~bmask2)]





    # utils.imshow(result)
    # plt.imshow(result, cmap='gray')  # 'gray' is the colormap for grayscale images

    # Show the plot
    # plt.show()

    return result

    # raise NotImplementedError()


def stitch_imgs(imgs: List[np.ndarray]) -> np.ndarray:
    """
    Stitch a list of images together into an image mosaic.
    imgs: list of images to be stitched together. You may assume the order
    the images appear in the list is the order in which they should be stitched
    together.

    Args:
        imgs (List[np.ndarray]): list of images to be stitched together. You may
        assume the order the images appear in the list is the order in which
        they should be stitched together.

    Returns:
        np.ndarray: the final, stitched image
    """
    
    numofimg = len(imgs)
    # print(numofimg)

    base_img = imgs[0]
    for i in range(1,numofimg):
        cur_img = imgs[i]
        
        src_pts, dest_pts = utils.sift_matches(base_img, cur_img)

        # Run RANSAC using parameters defined at the top of solutions.py
        inlier_idx, H = run_RANSAC(dest_pts,src_pts,CHALLENGE_1C_RANSAC_N,CHALLENGE_1C_RANSAC_EPS)
        corner_pts= np.array([[0,0],[0,cur_img.shape[0]],[cur_img.shape[1],cur_img.shape[0]],[cur_img.shape[1],0]])

        # print(corner_pts)
        transfer_corner_pts = apply_homography(H,corner_pts)
        # print(transfer_corner_pts)

        padding = [0,0,0,0] #[left,top,right,bot]
        padding = [0,base_img.shape[0],base_img.shape[1],0]
        img_size = [0,base_img.shape[0],base_img.shape[1],0]

        for pt in transfer_corner_pts:
            if pt[0] < padding[0]:
                padding[0] = pt[0]
            if pt[0] > padding[2]:
                padding[2] = pt[0]

            if pt[1] < padding[3]:
                padding[3] = pt[1]
            if pt[1] > padding[1]:
                padding[1] = pt[1]
        padding = np.array(padding).astype(int)
        # print(padding)
        pad_img = np.zeros((padding[1]-padding[3],padding[2]-padding[0],3))
        pad_img[(-padding[3]):(-padding[3]+base_img.shape[0]),(-padding[0]):(-padding[0]+base_img.shape[1])] = base_img
        mask1 = np.zeros((pad_img.shape[0],pad_img.shape[1]), dtype=bool)
        mask1[(-padding[3]):(-padding[3]+base_img.shape[0]),(-padding[0]):(-padding[0]+base_img.shape[1])] = True
        # utils.imshow(pad_img)

        base_img = pad_img
        h_pts = apply_homography(H,src_pts[inlier_idx,:])
        # src_pts[:,0] = src_pts[:,0]-padding[0]
        # src_pts[:,1] = src_pts[:,1]-padding[3]
        input_pts = src_pts[inlier_idx,:]

        h_pts[:,0] = h_pts[:,0]-padding[0]
        h_pts[:,1] = h_pts[:,1]-padding[3]
        # output_img = warp_img_onto(cur_img,base_img,input_pts,h_pts)
        # utils.imshow(output_img)
        # base_img = output_img

        H = compute_homography(input_pts, h_pts)
        H_inv = np.linalg.inv(H)
        # print(H_inv.shape)
        
        dest_height = base_img.shape[0]
        dest_width = base_img.shape[1]
        dest_canvas_width_height = dest_width,dest_height
        mask, img_warp = backward_warp_img(cur_img,H_inv, dest_canvas_width_height)
        # utils.imshow(mask1)
        # print(mask1.shape)
        # print(mask.shape)
        # print(mask1)
        mask = mask[:,:,1]
        # print(mask.shape)
        # print(mask)

        base_img = blend_image_pair(base_img,mask1,img_warp,mask,'blend')
        # utils.imshow(base_img)

    return base_img
    raise NotImplementedError()

def build_your_own_panorama() -> np.ndarray:
    """
    Build your own panorama using images in results/challenge_1f_input/.
    """
    input_path = utils.get_result_path("challenge_1f_input") # Do not change

    # Load images
    # TODO list your file names in the order they should be stitched
    file_names = ["room-center.jpg", "room-left.jpg", "room-right.jpg"]
    imgs = []
    for f_name in file_names:
        img_path = str((Path(input_path) / f_name).resolve())
        img = utils.imread(img_path, flag=None, rgb=True, normalize=True)
        imgs.append(img)

    return stitch_imgs(imgs)
    raise NotImplementedError()
