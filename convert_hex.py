import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt

image = []
image_path = "Set12/08.png"

def loadImage():
    global image
    # global image_path
    # image_path = input("input image path")
    image = np.array(utils.load_grayscale_image(image_path))
    # print(image.shape)

def visualize_image(image, name):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(name)
    plt.show()

def scale_Image(scaling_factor):
    global image
    m,n = image.shape
    scaled_image = np.empty((scaling_factor*m, scaling_factor*n))

    for i in range(m):
        for j in range(n):
            pixel_val = image[i,j]
            scaled_image[scaling_factor*i:scaling_factor*(i+1), scaling_factor*j:scaling_factor*(j+1)] = pixel_val
    return scaled_image


def get_square_strut():
    global image
    i, j, iteration = 0, 0, 1
    isFirst = True
    while True:
        image_height, image_width = image.shape
        end_i = min(i + 9, image_height)
        if(iteration%2==0 and isFirst):
            end_j = min(j + 4, image_width)
        else:
            end_j = min(j + 8, image_width)
        # Scan image from i to i+8 and j to j+7
        child_sq_mat = image[i:end_i, j:end_j]

        # Check if j has reached the end of the image width
        if j + 8 >= image_width:
            j = 0
            iteration +=1
            i += 7
        else:
            if(iteration%2==0 and isFirst):
                j += 4
                isFirst = False
            else:
                j += 8

def calc_avg(child_sq_mat):
    total_sum = 0
    if child_sq_mat.shape == (9, 8):
        total_sum += np.sum(image[[0, 8], 3:5])
        total_sum += np.sum(image[[1, 7], 1:6])
        total_sum += np.sum(image[2:7, :])
    elif child_sq_mat.shape == (9,4):
        total_sum += np.sum(image[[0, 8], 0])
        total_sum += np.sum(image[[1, 7], 0:3])
        total_sum += np.sum(image[2:7, :])
    else:
        total_sum = np.sum(image)
    return total_sum
    
if __name__=='__main__':
    loadImage()
    scaled_img = scale_Image(7)
    # print(image.shape)
    # print(scaled_img.shape)
    # visualize_image(image, "original")
    # visualize_image(scaled_img, "sclaed")