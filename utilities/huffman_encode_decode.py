import heapq
from collections import Counter, defaultdict
import struct

class Node:
    def __init__(self, value=None, freq=None, left=None, right=None):
        self.value = value
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def calculate_frequency(array):
    return Counter(array)

def construct_huffman_tree(frequencies):
    heap = [Node(value, freq) for value, freq in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    return heap[0]

def generate_huffman_codes(root):
    codes = {}

    def traverse(node, code):
        if node:
            if node.value is not None:
                codes[node.value] = code
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')

    traverse(root, '')
    return codes

def huffman_encode(array, codes):
    encoded_data = ''
    for value in array:
        encoded_data += codes[value]
    return encoded_data

def huffman_decode(encoded_data, root):
    decoded_data = []
    current_node = root
    for bit in encoded_data:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right
        if current_node.value is not None:
            decoded_data.append(int(current_node.value))
            current_node = root
    return decoded_data

# def write_to_file(filename, root, encoded_data):
#     with open(filename, 'wb') as f:
#         f.write(struct.pack('I', len(encoded_data)))
#         write_tree(f, root)
#         write_encoded_data(f, encoded_data)

def write_to_file(filename, encoded_data):
    with open(filename, 'wb') as f:
        f.write(struct.pack('I', len(encoded_data)))
        # write_tree(f, root)
        write_encoded_data(f, encoded_data)

def write_tree(f, node):
    if node:
        if node.value is not None:
            f.write(b'1')
            f.write(struct.pack('B', node.value))
        else:
            f.write(b'0')
            write_tree(f, node.left)
            write_tree(f, node.right)

def write_encoded_data(f, encoded_data):
    for i in range(0, len(encoded_data), 8):
        byte = encoded_data[i:i+8]
        f.write(bytes([int(byte, 2)]))

# def read_from_file(filename):
#     with open(filename, 'rb') as f:
#         encoded_data_length = struct.unpack('I', f.read(4))[0]
#         root = read_tree(f)
#         encoded_data = read_encoded_data(f, encoded_data_length)
#     return root, encoded_data

def read_from_file(filename):
    with open(filename, 'rb') as f:
        encoded_data_length = struct.unpack('I', f.read(4))[0]
        # root = read_tree(f)
        encoded_data = read_encoded_data(f, encoded_data_length)
    return encoded_data

def read_tree(f):
    bit = f.read(1)
    if not bit or bit == b'1':
        return Node(struct.unpack('B', f.read(1))[0], 0)
    else:
        left = read_tree(f)
        right = read_tree(f)
        return Node(None, 0, left, right)

def read_encoded_data(f, encoded_data_length):
    encoded_data = ''
    byte = f.read(1)
    while byte:
        encoded_data += f'{byte[0]:08b}'
        byte = f.read(1)
    # Use the length of the encoded data to remove any padding zeros
    last_byte_count = encoded_data_length % 8
    if last_byte_count == 0:
        return encoded_data
    last_byte_start = int(encoded_data_length / 8)
    return encoded_data[:last_byte_start*8] + encoded_data[-last_byte_count:]

if __name__=='__main__':
    array = [1, 2, 1, 3, 2, 1, 1, 4]
    frequencies = calculate_frequency(array)
    root = construct_huffman_tree(frequencies)
    huffman_codes = generate_huffman_codes(root)
    encoded_data = huffman_encode(array, huffman_codes)

    filename = 'test.bin'
    write_to_file(filename, huffman_codes, encoded_data)

    huffman_codes, encoded_data = read_from_file(filename)
    decoded_data = huffman_decode(encoded_data, huffman_codes)

    print("Original array:", array)
    print("Huffman codes:", huffman_codes)
    print("Encoded data:", encoded_data)
    print("Decoded data:", decoded_data)
