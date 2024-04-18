import numpy as np
import cv2
import huffman_encode_decode as huffman 
import struct

def load_grayscale_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def flatten_list(nested_list):
    flattened_list = []
    for i in nested_list:
        if isinstance(i, list):
            flattened_list.extend(flatten_list(i))
        else:
            flattened_list.append(i)
    return flattened_list

def divide_image_into_blocks(image, block_shape=(4, 8)):
    rows, cols = image.shape
    block_rows, block_cols = block_shape

    # Calculate the number of blocks in rows and columns
    num_blocks_rows = int(rows / block_rows) + (0 if rows % block_rows == 0 else 1)
    num_blocks_cols = int(cols / block_cols) + (0 if cols % block_cols == 0 else 1)

    if rows % block_rows != 0 or cols % block_cols != 0:
        pad_rows = block_rows - (rows % block_rows)
        pad_cols = block_cols - (cols % block_cols)
        image = np.pad(image, ((0, pad_rows), (0, pad_cols)), mode="constant")

    # Initialize an empty list to store the blocks
    blocks = []

    # Iterate over rows and columns to extract blocks
    for i in range(num_blocks_rows):
        for j in range(num_blocks_cols):
            block = image[i * block_rows: (i + 1) * block_rows, j * block_cols: (j + 1) * block_cols]
            blocks.append(flatten_list(block))

    return blocks

def combine_blocks_into_image(blocks, image_shape):
    blocks_array = np.array(blocks)
    num_blocks, num_blocks_rows, num_blocks_cols = blocks_array.shape
    image_rows, image_cols = image_shape
    block_count_col = int(image_rows / num_blocks_rows) + (0 if image_rows % num_blocks_rows == 0 else 1)
    block_count_row = int(image_cols / num_blocks_cols) + (0 if image_cols % num_blocks_cols == 0 else 1) 
    combined_image_height = block_count_col * num_blocks_rows
    combined_image_width = block_count_row * num_blocks_cols 

    # Initialize an empty array to store the combined image
    combined_image = np.zeros((combined_image_height, combined_image_width), dtype=np.uint8)

    # Iterate over rows and columns to combine blocks into the image
    count = 0
    for i in range(block_count_col):
        for j in range(block_count_row):
            combined_image[i*num_blocks_rows:(i+1)*num_blocks_rows, j*num_blocks_cols:(j+1)*num_blocks_cols] = blocks_array[count]
            count+=1

    # Trim the combined image to the original image shape
    combined_image = combined_image[:image_rows, :image_cols]

    return combined_image

def imshow(image):
    # Get the screen resolution
    screen_width, screen_height = 1920, 1080  # Change these values according to your screen resolution

    # Get the image dimensions
    print(image.shape)
    image_height, image_width = image.shape[:2]
    if image_height > screen_height  or image_width > screen_width:
        print('resized')
        # Determine the scaling factor to fit the image within the screen
        scale_factor = min(screen_width / image_width, screen_height / image_height)

        # Resize the image to fit within the screen while maintaining the aspect ratio
        resized_width = int(image_width * scale_factor)
        resized_height = int(image_height * scale_factor)
        resized_image = cv2.resize(image, (resized_width, resized_height))

        # Display the resized image
        cv2.imshow('Image', resized_image)
    else:    
        cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def huffman_encode(array):
    frequencies = huffman.calculate_frequency(array)
    root = huffman.construct_huffman_tree(frequencies)
    huffman_codes = huffman.generate_huffman_codes(root)
    encoded_data = huffman.huffman_encode(array, huffman_codes)
    return root, encoded_data

def huffman_decode(root, encoded_data):
    return huffman.huffman_decode(encoded_data, root)


