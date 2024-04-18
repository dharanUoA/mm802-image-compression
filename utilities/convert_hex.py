import utils
import numpy as np
import matplotlib.pyplot as plt

def scale_Image(image, scaling_factor):
    m,n = image.shape
    scaled_image = np.empty((scaling_factor*m, scaling_factor*n))

    for i in range(m):
        for j in range(n):
            pixel_val = image[i,j]
            scaled_image[scaling_factor*i:scaling_factor*(i+1), scaling_factor*j:scaling_factor*(j+1)] = pixel_val
    return scaled_image

def visualize_image(image, name):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(name)
    plt.show()
  
def hexagonal_lattice_transform(image, block_shape=(9, 8)):
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
    
    transformed_image = []
    min_cols = np.min([len(final_image[i]) for i in range(len(final_image))])
    for i in range(len(final_image)):
        transformed_image.append(final_image[i][0:min_cols])
    
    return transformed_image

def get_error(image, hex_image):
    m, n = hex_image.shape
    error = np.sqrt(np.sum(np.square(image[0:m,0:n] - hex_image))) / (m*n)
    return error     

if __name__=='__main__':
    image = np.array(utils.load_grayscale_image('image-datasets/Set12/12.png'))
    
    transformed_image = hexagonal_lattice_transform(image)
    
    visualize_image(transformed_image, "Image after transform")