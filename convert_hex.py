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

def scale_Image(image, scaling_factor):
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
    
def method(image, block_shape=(9, 8)):
    block_rows, block_cols = block_shape
    image = scale_Image(image, 7)
    rows, cols = image.shape

    i, j = 0, 0
    isEven = True
    final_image = []
    while i+9 < rows:
        row_block = []
        while j < cols:
            total_sum = 0
            count = 0
            if isEven:
                last_index = cols if j+8 > cols else j
                total_sum += np.sum(image[[i, i+8], last_index+3:last_index+5])
                total_sum += np.sum(image[[i+1, i+7], last_index+1:last_index+6])
                total_sum += np.sum(image[i+2:i+7, last_index:last_index+8])
                count = 56
                j += block_cols
            else:
                if j == 0:
                    last_index = cols if j+4 > cols else j
                    total_sum += np.sum(image[[i, i+8], last_index])
                    total_sum += np.sum(image[[i+1, i+7], last_index:last_index+3])
                    total_sum += np.sum(image[i+2:i+7, last_index:last_index+4])
                    j += int(block_cols / 2)
                    count = 28
                else:
                    last_index = cols if j+8 > cols else j
                    total_sum += np.sum(image[[i, i+8], last_index+3:last_index+5])
                    total_sum += np.sum(image[[i+1, i+7], last_index+1:last_index+6])
                    total_sum += np.sum(image[i+2:i+7, last_index:last_index+8])
                    j += block_cols
                    count = 56
            row_block.append(int(total_sum/count))
        isEven = not isEven
        i += (block_rows - 2)
        j = int(0)
        final_image.append(row_block)
    return final_image
    

if __name__=='__main__':
    image = np.array(utils.load_grayscale_image('Set12/01.png'))
    print(image.shape)
    # image_blocks = np.array([[162, 162, 162, 250, 161, 157, 163, 163],
    #                         [162, 162, 162, 162, 161, 157, 163, 163],
    #                         [162, 162, 162, 162, 161, 157, 163, 163],
    #                         [162, 162, 162, 162, 161, 157, 163, 163],
    #                         [162, 162, 162, 162, 161, 157, 163, 163],
    #                         [162, 162, 162, 162, 161, 157, 163, 163],
    #                         [162, 162, 162, 162, 161, 157, 163, 163],
    #                         [162, 162, 162, 162, 161, 157, 163, 163],
    #                         [162, 162, 162, 162, 161, 157, 163, 163]])
    
    final_image = method(image)
    # print(final_image)
    
    final_final_image = []
    for i in range(len(final_image)):
        block = []
        if i % 2 != 0:
            block = final_image[i][0:len(final_image[i])-1]
        else:
            block = final_image[i]
        final_final_image.append(block)

    # print(final_final_image)
    # print(image.shape)
    # print(scaled_img.shape)
    visualize_image(final_final_image, "original")
    print(np.array(final_final_image).shape)
    # visualize_image(scaled_img, "sclaed")