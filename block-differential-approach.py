import numpy as np
import utils
import huffman_encode_decode as huffman

BLOCK_SHAPE = (4, 8)

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
        # pixels = []
        # for i in range(4):
        #     for j in range(8):
        #         if (i+j) % 2 != 0:
        #             pixels.append(block[i, j])
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
    data = ''
    for i in range(len(encoded_blocks)):
        block = encoded_blocks[i]
        data += ('1' if block.is_encoded else '0')
        if block.is_encoded:
            data += format(block.max, '08b')
            data += format(block.bits_per_value, '03b')
            format_string = '0' + str(block.bits_per_value) + 'b'
            data += ''.join([format(j, format_string) for j in block.block])
        else: 
            data += ''.join([format(j, '08b') for j in block.block])

    return [int(i) for i in data]

def bitstream_to_encoded_blocks(bitstream):
    encoded_blocks = []
    i = 0
    ITEMS_IN_BLOCK = BLOCK_SHAPE[0] * BLOCK_SHAPE[1]
    while i < len(bitstream):
        if int(bitstream[i]) == 1: # block encoded
            i+=1
            max = int(''.join(map(str, bitstream[i:i+8])), 2)
            i+=8
            bits_per_value = int(''.join(map(str, bitstream[i:i+3])), 2)
            i+=3
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

if __name__ == '__main__':
    image = utils.load_grayscale_image("Set12/02.png")
    image_shape = image.shape
    
    image_blocks = utils.divide_image_into_blocks(image, BLOCK_SHAPE)

    encoded_blocks = apply_block_differential_encoding(image_blocks)
    bit_stream_array = convert_encoded_block_to_bitstream(encoded_blocks)
    root, encoded_data = utils.huffman_encode(bit_stream_array)

    file_name = 'block_differential.huf'
    huffman.write_to_file(file_name, root, encoded_data)

    root, encoded_data = huffman.read_from_file(file_name)
    decoded_bitstream = utils.huffman_decode(root, encoded_data)
    encoded_blocks = bitstream_to_encoded_blocks(decoded_bitstream)
    reconstructed_image_blocks = apply_block_differential_decoding(encoded_blocks)

    print_error(image_blocks, reconstructed_image_blocks)

    reconstructed_image = utils.combine_blocks_into_image(reshape_to_original_block_size(reconstructed_image_blocks), image_shape)
    utils.imshow(reconstructed_image)
