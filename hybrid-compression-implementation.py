import numpy as np
import cv2
import utilities.huffman_encode_decode as huffman

class BlockDifferenrial:
    block = None
    max = None
    bits_per_value = None

    def __init__(self, is_encoded, block=None):
        self.is_encoded = is_encoded
        self.block = block
        
def load_grayscale_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def get_msb_and_lsb_nibbles(number):
    # Get the most significant nibble (MSB)
    msb = (number >> 4) & 0xF
    
    # Get the least significant nibble (LSB)
    lsb = number & 0xF
    
    return msb, lsb

def apply_mapping_transform(image_blocks):
    transformed_half = []
    for block in image_blocks:
        pixels = []
        for i in range(4):
            for j in range(8):
                if (i+j) % 2 == 0:
                    pixels.append(block[i, j])

        transform_table = np.zeros([16, 16], dtype=int)
        for index, pixel in enumerate(pixels):
            msb, lsb = get_msb_and_lsb_nibbles(pixel)
            transform_table[index, msb] = 1
            transform_table[lsb, index] = 1
        output_block = np.zeros((4, 8), dtype=int)
        for col in range(16):
            binary_numbers = ["".join(str(transform_table[row + x, col]) for x in range(8)) for row in range(0, 16, 8)]
            decimal_values = [int(binary_number, 2) for binary_number in binary_numbers]
            output_block[int(col / 4), (col % 4)*2:((col % 4)*2)+2] = decimal_values
        
        transformed_half.append(output_block)
    return transformed_half

def apply_reverse_mapping_transform(transformed_blocks):
    original_blocks = []
    for block in transformed_blocks:
        transform_table = np.zeros([16, 16])
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                element = block[i, j]
                binary_string = format(element, '08b')
                row_index = 0 if j % 2 == 0 else block.shape[1]
                col_index = (i * int(block.shape[1]/2)) + int(np.floor(j/2))
                transform_table[row_index:row_index+block.shape[1], col_index] = np.transpose([int(bit) for bit in binary_string])
        row_sum = np.sum(transform_table, axis=1)
        col_sum = np.sum(transform_table, axis=0)
        max_row_sum_index = np.argmax(row_sum)
        max_col_sum_index = np.argmax(col_sum)
        msb_dict = {}
        nibble_index = None
        for index, value in enumerate(row_sum):
            nibble_indexes = np.where(transform_table[index, :] == 1)[0]
            if value == 1:
                nibble_index = nibble_indexes[0]
            elif value > 1:
                nibble_index = nibble_indexes[np.argmin([np.abs(i-max_col_sum_index) for i in nibble_indexes])]
            else:
                nibble_index = 0
            msb_nibble = format(nibble_index, '04b')
            msb_dict[index] = msb_nibble
            nibble_index = None
        lsb_dict = {}
        nibble_index = None
        for index, value in enumerate(col_sum):
            nibble_indexes = np.where(transform_table[:, index] == 1)[0]
            if value == 1:
                nibble_index = nibble_indexes[0]
            elif value > 1:
                nibble_index = nibble_indexes[np.argmin([np.abs(i-max_row_sum_index) for i in nibble_indexes])]
            else:
                nibble_index = 0
            lsb_nibble_pred_1 = format(nibble_index, '04b')
            lsb_nibble = lsb_nibble_pred_1
            lsb_dict[index] = lsb_nibble
            nibble_index = None
        arr = []
        for i in range(16):
            arr.append(int(msb_dict[i] + lsb_dict[i], 2))
        # original_blocks.append(np.reshape(arr, [4,4]))
        original_blocks.append([arr])
    return original_blocks

def apply_block_differential_encoding(image_blocks):
    threashold = 127
    encoded_half = []
    for block in image_blocks:
        pixels = []
        for i in range(4):
            for j in range(8):
                if (i+j) % 2 != 0:
                    pixels.append(block[i, j])
        max = np.max(pixels)
        min = np.min(pixels)
        if max - min < threashold:
            block_differential = BlockDifferenrial(True)
            pixels[:] = max - pixels
            block_differential.max = max
            new_min = np.min(pixels)
            new_max = np.max(pixels)
            bit_required = int(new_max).bit_length()
            block_differential.bits_per_value = bit_required
            np.clip(pixels, new_min, new_max)
            block_differential.block = pixels
        else:
            block_differential = BlockDifferenrial(False, pixels)
        encoded_half.append(block_differential)
    return encoded_half

def apply_block_differential_decoding(encoded_blocks):
    original_blocks = []
    for block in encoded_blocks:
        if block.is_encoded == 1:
            original_blocks.append([block.max - x for x in block.block])
        else:
            original_blocks.append([block.block])
    return original_blocks

def reconstruct_image(transformed_half, encoded_half, shape):
    # Placeholder: Replace with actual implementation
    reconstructed_image = []
    for i in range(len(transformed_half)):
        first_half = np.array(transformed_half[i]).reshape(4, 4)
        second_half = np.array(encoded_half[i]).reshape(4, 4)
        reconstructed_block = np.zeros((4, 8))
        for j in range(4):
            for k in range(0,8,2):
                if j%2==0: # 0, 2
                    reconstructed_block[j,k] = first_half[j,int(k/2)]
                    reconstructed_block[j,k+1] = second_half[j,int(k/2)]
                else: # 1, 3
                    reconstructed_block[j,k+1] = first_half[j,int(k/2)]
                    reconstructed_block[j,k] = second_half[j,int(k/2)]
        reconstructed_image.append(reconstructed_block)
    combined_image = combine_blocks_into_image(reconstructed_image, shape)
    return combined_image

def divide_image_into_blocks(image, block_shape=(4, 8)):
    """
    Divides the input grayscale image into blocks of the specified shape.

    Args:
        image (numpy.ndarray): Grayscale image as a numpy array.
        block_shape (tuple): Desired block shape (rows, columns).

    Returns:
        List of numpy arrays, each representing a block.
    """
    rows, cols = image.shape
    block_rows, block_cols = block_shape

    # Calculate the number of blocks in rows and columns
    num_blocks_rows = rows // block_rows
    num_blocks_cols = cols // block_cols

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
            blocks.append(block)

    return blocks

def combine_blocks_into_image(blocks, image_shape):
    # image_shape = (4, 40)
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
    image_height, image_width = image.shape[:2]
    if image_height > screen_height  or image_width > screen_width:
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

image = load_grayscale_image("image-datasets/Set12/12.png")

image_shape = image.shape

block_shape = (4, 8)

image_blocks = divide_image_into_blocks(image, block_shape)

transformed_first_half = apply_mapping_transform(image_blocks)
root, encoded_data  = huffman_encode(np.array(transformed_first_half).flatten())

decoded_data = huffman_decode(root, encoded_data)

count = int(len(decoded_data) / 32)
transformed_first_half = np.reshape(np.array(decoded_data), (count, 4, 8))

encoded_second_half = apply_block_differential_encoding(image_blocks)
data = ''
for i in range(len(encoded_second_half)):
    block = encoded_second_half[i]
    data += ('1' if block.is_encoded else '0')
    if block.is_encoded:
        data += format(block.max, '08b')
        data += format(block.bits_per_value, '03b')
        format_string = '0' + str(block.bits_per_value) + 'b'
        data += ''.join([format(j, format_string) for j in block.block])
    else: 
        data += ''.join([format(j, '08b') for j in block.block])

list = [int(i) for i in data]

root, encoded_data = huffman_encode(np.array(list).flatten())

decoded_data = huffman_decode(root, encoded_data)

encoded_second_half = []
i = 0
while i < len(decoded_data):
    if int(decoded_data[i]) == 1: # block encoded
        i+=1
        max = int(''.join(map(str, decoded_data[i:i+8])), 2)
        i+=8
        bits_per_value = int(''.join(map(str, decoded_data[i:i+3])), 2)
        i+=3
        block = [int(''.join(map(str, decoded_data[i+j:i+j+bits_per_value])), 2) for j in range(0,bits_per_value*16, bits_per_value)]
        i+=(bits_per_value*16)
        block_obj = BlockDifferenrial(True, block)
        block_obj.max = max
        block_obj.bits_per_value = bits_per_value
        encoded_second_half.append(block_obj)
    else: # block not encoded
        i+=1
        block = [int(''.join(map(str, decoded_data[i+j:i+j+8])), 2) for j in range(0,8*16, 8)]
        i+=(8*16)
        block_obj = BlockDifferenrial(False, block)
        encoded_second_half.append(block_obj)

original_first_half = apply_reverse_mapping_transform(transformed_first_half)

error = 0
for k in range(len(original_first_half)):
    block = image_blocks[k]
    pixels = []
    for i in range(4):
        for j in range(8):
            if (i+j) % 2 == 0:
                pixels.append(block[i, j])
    error += np.sqrt(np.sum(np.square(np.array(pixels) - original_first_half[k])))

original_second_half = apply_block_differential_decoding(encoded_second_half)

error = 0
for k in range(len(original_second_half)):
    block = image_blocks[k]
    pixels = []
    for i in range(4):
        for j in range(8):
            if (i+j) % 2 != 0:
                pixels.append(block[i, j])
    error += np.sqrt(np.sum(np.square(np.array(pixels) - original_second_half[k])))

reconstructed_image = reconstruct_image(original_first_half, original_second_half, image_shape)

error = np.sqrt(np.sum(np.square(image - reconstructed_image)))
imshow(reconstructed_image)