import numpy as np
import utilities.utils as utils
import utilities.huffman_encode_decode as huffman
import os
import imagecodecs
import csv
import utilities.convert_hex as hex
import cv2

BLOCK_SHAPE = (8, 8)

class BlockDifferenrial:
    block = None
    max = None
    bits_per_value = None

    def __init__(self, is_encoded, block=None):
        self.is_encoded = is_encoded
        self.block = block

def apply_block_differential_encoding(image_blocks):
    threashold = 127
    encoded_half = []
    for block in image_blocks:
        pixels = np.array(block).flatten()
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
            original_blocks.append(block.block)
    return original_blocks

def convert_encoded_block_to_bitstream(encoded_blocks):
    bitstream = []
    for block in encoded_blocks:
        bitstream.append(1 if block.is_encoded else 0)
        if block.is_encoded:
            bitstream.extend([int(b) for b in format(block.max, '08b')])
            bitstream.extend([int(b) for b in format(block.bits_per_value, '03b')])
            if block.bits_per_value != 0:
                format_string = '0' + str(block.bits_per_value) + 'b'
                bitstream.extend([int(b) for j in block.block for b in format(j, format_string)])
        else: 
            bitstream.extend([int(b) for j in block.block for b in format(j, '08b')])

    return bitstream

def bitstream_to_encoded_blocks(bitstream):
    encoded_blocks = []
    i = 0
    ITEMS_IN_BLOCK = BLOCK_SHAPE[0] * BLOCK_SHAPE[1]
    while i < len(bitstream):
        if bitstream[i] == 1: # block encoded
            i+=1
            max = int(''.join(map(str, bitstream[i:i+8])), 2)
            i+=8
            bits_per_value = int(''.join(map(str, bitstream[i:i+3])), 2)
            i+=3
            block = []
            if bits_per_value == 0:
                block = [0 for i in range(ITEMS_IN_BLOCK)]
            else:
                block = [int(''.join(map(str, bitstream[i+j:i+j+bits_per_value])), 2) for j in range(0,bits_per_value*ITEMS_IN_BLOCK, bits_per_value)]
                i+=(bits_per_value*ITEMS_IN_BLOCK)
            block_obj = BlockDifferenrial(True, block)
            block_obj.max = max
            block_obj.bits_per_value = bits_per_value
            encoded_blocks.append(block_obj)
        else: # block not encoded
            i+=1
            block = [int(''.join(map(str, bitstream[i+j:i+j+8])), 2) for j in range(0,8*ITEMS_IN_BLOCK, 8)]
            i+=(8*ITEMS_IN_BLOCK)
            block_obj = BlockDifferenrial(False, block)
            encoded_blocks.append(block_obj)
    return encoded_blocks

def reshape_to_original_block_size(arr):
    arr = np.array(arr)
    reshaped_arr = arr.reshape(arr.shape[0], BLOCK_SHAPE[0], BLOCK_SHAPE[1])
    return reshaped_arr.tolist()

def print_error(image_blocks, reconstructed_image_blocks):
    error = 0
    for k in range(len(reconstructed_image_blocks)):
        block = image_blocks[k]
        block = np.array(block).flatten()
        error += np.sqrt(np.sum(np.square(block - reconstructed_image_blocks[k])))
    print('error', error)

def get_file_size(relative_file_path):
    absolute_file_path = os.path.join(os.getcwd(), relative_file_path)
    return os.path.getsize(absolute_file_path)

def write_jpeg_ls(filename, bytes):
    with open(filename, 'wb') as f:
        f.write(bytes)

def image_entropy(image):
    # Convert the image to grayscale if it's a color image
    # Compute histogram of pixel intensity values
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    
    # Normalize histogram to obtain probabilities
    hist_norm = hist / np.sum(hist)
    
    # Calculate entropy
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))  # Add a small value to prevent log(0)
    
    return entropy

def calculate_psnr(original_img, compressed_img):
    max_pixel = 255.0  # Assuming pixel values range from 0 to 255
    mse = np.mean((original_img - compressed_img) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


if __name__ == '__main__':
    # FILES = [f"{i:02}.png" for i in range(1, 13)]
    # FILES = [f"{i}.jpg" for i in range(0, 3)]
    # data = [["Name","Size Using Noval Approach", "Size Using JPEG-LS", "MSE", "Image Entropy", "Compressed Image Entropy", "PSNR"]]
    # for file in FILES:
    file = '01.png'
    image_path = f"image-datasets/Set12/{file}"
    image = utils.load_grayscale_image(image_path)
    original_image = utils.load_grayscale_image(image_path)

    entropy1 = image_entropy(image)

    transformed_image = hex.hexagonal_lattice_transform(image)

    image = np.array(transformed_image, dtype='uint8')
    entropy2 = image_entropy(image)
    image_shape = image.shape
    

    image_blocks = utils.divide_image_into_blocks(image, BLOCK_SHAPE)
    encoded_blocks = apply_block_differential_encoding(image_blocks)
    bit_stream_array = convert_encoded_block_to_bitstream(encoded_blocks)
    
    encoded_data = ''.join(map(str, bit_stream_array))

    # file_name = 'block-differential.bin'
    # huffman.write_to_file(file_name, encoded_data)
    # encoded_data = huffman.read_from_file(file_name)
    
    encoded_bit_stream = [int(i) for i in encoded_data]

    encoded_blocks = bitstream_to_encoded_blocks(encoded_bit_stream)

    reconstructed_image_blocks = apply_block_differential_decoding(encoded_blocks)

    reconstructed_image = utils.combine_blocks_into_image(reshape_to_original_block_size(reconstructed_image_blocks), image_shape)
    utils.imshow(reconstructed_image)
    
    # hex_error = hex.get_error(original_image, reconstructed_image)
    # compressed_file_size = get_file_size(file_name)
    # compressed_data = imagecodecs.jpegls_encode(image)
    # jpeg_ls_filename = 'jpeg-ls.bin'
    # write_jpeg_ls(jpeg_ls_filename, compressed_data)
    # image_size = get_file_size(jpeg_ls_filename)
    # psnr = calculate_psnr(image, reconstructed_image)
    # data.append([file, compressed_file_size, image_size, hex_error, entropy1, entropy2, psnr])