import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


#--------------------------------------------------------------------------
# Academic Honesty Policy
#--------------------------------------------------------------------------
def sign_academic_honesty_policy():
    print("I, %s (%s), certify that I have read and agree to the Code of Academic Integry." % ('Wei Zhang', 'wz2580'))


#--------------------------------------------------------------------------
# Challenge 1: Optical flow using Lucas-Kanade
#--------------------------------------------------------------------------
def computeFlow(img1: np.ndarray, img2: np.ndarray, window_size: int) -> np.ndarray:
    """
    Use Lucas-Kanade method to compute the optical flow between two images.

    At each pixel, compute the optical flow (u, v) between the two images. u
    is the flow in the x (column) direction, and v is the flow in the y (row)
    direction.

    Args:
        img1 (np.ndarray): the first image (HxW)
        img2 (np.ndarray): the second image (HxW)
        window_size (int): size of the window within which optical flow is constant

    Returns:
        np.ndarray: calculated optical flow (u, v) at each pixel (HxWx2)
    """
    pad = window_size//2
    r = len(img1)
    c = len(img1[0])
    img1_padded = np.pad(img1, ((0,0),(0,1)), mode='constant')
    img2_padded = np.pad(img2, ((0,0),(0,1)), mode='constant')
    img1_padded1 = np.pad(img1, ((0,1),(0,0)), mode='constant')
    img2_padded1 = np.pad(img2, ((0,1),(0,0)), mode='constant')

    # s(img1_padded)
    # s(img2_padded)

    # compute Ix, Iy, It
    # Ix = 1/4 (sum x=k+1) - 1/4 (sum of x = k), for y =l,l+1 and t= t and t+1
    # Iy = 1/4 (sum y=l+1) - 1/4 (sum of y = l) 
    # It = 1/4 (sum t=t) - 1/4 (sum of t = t+1)

    Ix1 = img1_padded[:,1:] - img1
    Ix2 = img2_padded[:,1:] - img2
    Iy1 = img1_padded1[1:,:] - img1
    Iy2 = img2_padded1[1:,:] - img2
    It = img2 - img1

    # pad them for calculation
    Ix1_padded = np.pad(Ix1, ((0, 0), (0, 1)), mode='constant')
    Ix2_padded = np.pad(Ix2, ((0, 0), (0, 1)), mode='constant')
    Iy1_padded = np.pad(Iy1, ((0, 1), (0, 0)), mode='constant')
    Iy2_padded = np.pad(Iy2, ((0, 1), (0, 0)), mode='constant')
    It_padded = np.pad(It, ((0, 1), (0, 1)), mode='constant')

    # It_padded = It_padded[:r+1,1:]
    # Kernels for computing gradients
    kernel_x = np.array([[1, 1]]) * 0.25
    kernel_y = np.array([[1], [1]]) * 0.25
    kernel_t = np.ones((2,2)) * 0.25

    I_x_pixel = signal.convolve2d(Ix1_padded, kernel_x, mode='valid')+signal.convolve2d(Ix2_padded, kernel_x, mode='valid')
    I_y_pixel = signal.convolve2d(Iy1_padded, kernel_y, mode='valid')+signal.convolve2d(Iy2_padded, kernel_y, mode='valid')
    I_t_pixel = signal.convolve2d(It_padded, kernel_t, mode='valid')

    IxIx = I_x_pixel*I_x_pixel
    IxIy = I_x_pixel*I_y_pixel
    IyIy = I_y_pixel*I_y_pixel
    IxIt = I_x_pixel*I_t_pixel
    IyIt = I_y_pixel*I_t_pixel

    IxIx_padded = np.pad(IxIx, pad, mode='constant')
    IxIy_padded = np.pad(IxIy, pad, mode='constant')
    IyIy_padded = np.pad(IyIy, pad, mode='constant')
    IxIt_padded = np.pad(IxIt, pad, mode='constant')
    IyIt_padded = np.pad(IyIt, pad, mode='constant')

    kernel = np.ones((window_size,window_size))
    IxIx_sum = signal.convolve2d(IxIx_padded, kernel, mode='valid')
    IxIy_sum = signal.convolve2d(IxIy_padded, kernel, mode='valid')
    IyIy_sum = signal.convolve2d(IyIy_padded, kernel, mode='valid')
    IxIt_sum = signal.convolve2d(IxIt_padded, kernel, mode='valid')
    IyIt_sum = signal.convolve2d(IyIt_padded, kernel, mode='valid')

    u_output = np.zeros((r,c,2))
    for i in range(r):
        for j in range(c):
            ATA = np.zeros((2,2))
            ATA[0,0]= IxIx_sum[i,j]
            ATA[0,1]= IxIy_sum[i,j]
            ATA[1,0]= IxIy_sum[i,j]
            ATA[1,1]= IyIy_sum[i,j]

            ATB = np.zeros((2,1))
            ATB[0,0] = -IxIt_sum[i,j]
            ATB[1,0] = -IyIt_sum[i,j]
            try:
                u = np.linalg.inv(ATA) @ ATB
                u_output[i, j] = u.reshape((1, 2))
            except np.linalg.LinAlgError:
                # If the matrix is singular, assign [0, 0] to u_output[i, j]
                u_output[i, j] = [0, 0]
                # print("singluar matrix")
            # u = np.linalg.inv(ATA) @ ATB
            # u_output[i,j] = u.reshape((1,2))
        # print(f"{i =}")


    # I_x_pixel_padded = np.pad(I_x_pixel, pad, mode='constant')
    # I_y_pixel_padded = np.pad(I_y_pixel, pad, mode='constant')
    # I_t_pixel_padded = np.pad(I_t_pixel, pad, mode='constant')


    # compute A = [[Ix11 Iy11] [Ix12 Iy12]... [Ixnn Iynn ]]
    # compute B = [[It11] [It12]... [Itnn]]
    # compute u = (A.t @ A)^(-1) @ A.T @ B

    # hint from TA : Consider breaking your computation of the partial derivatives (slide 9 in the optical flow slides) into three 2D kernels that represent the partial w.r.t. 
    # x and y and t then convolve your images with these using signal.convolve2d (scipy.signal is already imported). This should speed up your computation considerably if you are currently using a for loop.


    # print(f"{r =}")
    # print(f"{c =}")
    # print(f"{img1.shape =}")
    # print(f"{img2.shape =}")
    # print(f"{img1_padded.shape =}")
    # print(f"{img2_padded.shape =}")
    # print(f"{Ix1.shape =}")
    # print(f"{Ix2.shape =}")
    # print(f"{Iy1.shape =}")
    # print(f"{Iy2.shape =}")
    # print(f"{It.shape =}")
    # print(f"{I_t_pixel.shape =}")
    # print(f"{pad =}")
    # print(f"{I_x_pixel.shape =}")
    # print(f"{I_t_pixel.shape =}")
    # print(f"{I_y_pixel.shape =}")
    # print(f"{IxIx_sum.shape =}")
    # print(f"{IxIy_sum.shape =}")
    # print(f"{IyIy_sum.shape =}")
    # print(f"{IxIt_sum.shape =}")
    # print(f"{IyIt_sum.shape =}")
    # print(f"{ATA =}")
    # print(f"{ATB =}")
    # print(f"{u =}")

    # s(img1)
    # s(img2)
    # s(img1_padded)
    # s(img2_padded)
    # s(It_padded)
    return u_output


def s(x):
    plt.imshow(x)
    plt.show()