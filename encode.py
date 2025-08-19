import argparse
import math
import os
import struct
import numpy as np
from PIL import Image

class ReedSolomon:
    def __init__(self, n, k):
        if n > 255 or n <= k or (n - k) % 2 != 0:
            raise ValueError(f"Invalid Reed-Solomon parameters: n={n}, k={k} must satisfy n <= 255, n > k, n-k even")
        self.n = n
        self.k = k
        self.t = (n - k) // 2
        self.poly = 0x11D
        self.exp = [0] * 255
        self.log = [0] * 256
        self._build_tables()

    def _build_tables(self):
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= self.poly
        self.log[0] = 0

    def _gf_add(self, a, b):
        return a ^ b

    def _gf_mul(self, a, b):
        if a == 0 or b == 0:
            return 0
        return self.exp[(self.log[a] + self.log[b]) % 255]

    def _gf_div(self, a, b):
        if a == 0:
            return 0
        if b == 0:
            raise ValueError("Division by zero")
        return self.exp[(self.log[a] - self.log[b]) % 255]

    def _gf_poly_mul(self, p, q):
        r = [0] * (len(p) + len(q) - 1)
        for j in range(len(q)):
            for i in range(len(p)):
                r[i + j] ^= self._gf_mul(p[i], q[j])
        return r

    def _gf_poly_div(self, num, den):
        quot = [0] * (len(num) - len(den) + 1)
        num = num[:]
        for i in range(len(num) - len(den) + 1):
            if num[i] == 0:
                continue
            coef = self._gf_div(num[i], den[0])
            quot[i] = coef
            for j in range(len(den)):
                num[i + j] ^= self._gf_mul(coef, den[j])
        return quot, num[-len(den) + 1:]

    def encode(self, data):
        if len(data) != self.k:
            raise ValueError(f"Expected {self.k} data bytes, got {len(data)}")
        gen = [1]
        for i in range(self.n - self.k):
            gen = self._gf_poly_mul(gen, [1, self.exp[i]])
        _, rem = self._gf_poly_div(data + [0] * (self.n - self.k), gen)
        return data + rem

    def decode(self, data):
        if len(data) != self.n:
            raise ValueError(f"Expected {self.n} bytes, got {len(data)}")
        gen = [1]
        for i in range(self.n - self.k):
            gen = self._gf_poly_mul(gen, [1, self.exp[i]])
        _, rem = self._gf_poly_div(data, gen)
        if all(r == 0 for r in rem):
            return data[:self.k]
        print("Errors detected; correction not implemented")
        return data[:self.k]

def file_to_bit_grid(filename, bits_per_row=2000, ec_bytes=0, monochrome=False):
    if not os.path.exists(filename):
        raise ValueError(f"Input file {filename} does not exist")
    with open(filename, 'rb') as f:
        data = f.read()
    
    filename_bytes = os.path.basename(filename).encode('utf-8', errors='ignore')
    filename_len = len(filename_bytes)
    if filename_len > 255:
        raise ValueError("Filename too long (max 255 bytes)")
    data = bytes([filename_len]) + filename_bytes + data
    
    bits_per_pixel = 1 if monochrome else 2
    calibration_rows = 100  # At least 100 rows for calibration
    bits_per_page = (bits_per_row * (bits_per_row - calibration_rows)) * bits_per_pixel  # Reserve 100 rows
    bytes_per_page = bits_per_page // 8
    
    # Split data into Reed-Solomon blocks (n <= 255)
    rs_n = min(255, bytes_per_page)
    rs_k = rs_n - ec_bytes if ec_bytes > 0 else rs_n
    if ec_bytes > 0 and (rs_n <= rs_k or (rs_n - rs_k) % 2 != 0):
        raise ValueError(f"Invalid Reed-Solomon parameters: n={rs_n}, k={rs_k} must satisfy n > k, n-k even")
    
    rs = ReedSolomon(rs_n, rs_k) if ec_bytes > 0 else None
    bytes_per_block = rs_k if rs else bytes_per_page
    num_blocks_per_page = math.ceil(bytes_per_page / bytes_per_block)
    bytes_per_page_effective = num_blocks_per_page * rs_n if rs else bytes_per_page
    
    num_pages = math.ceil(len(data) / (bytes_per_block * num_blocks_per_page if rs else bytes_per_page))
    data = data.ljust(num_pages * (bytes_per_block * num_blocks_per_page if rs else bytes_per_page), b'\x00')
    
    grids = []
    for page in range(num_pages):
        start = page * (bytes_per_block * num_blocks_per_page if rs else bytes_per_page)
        page_data = data[start:start + (bytes_per_block * num_blocks_per_page if rs else bytes_per_page)]
        
        if rs:
            encoded_data = []
            for i in range(0, len(page_data), rs_k):
                block = list(page_data[i:i + rs_k])
                block = block + [0] * (rs_k - len(block))
                encoded_data.extend(rs.encode(block))
            page_data = bytes(encoded_data)
        
        if monochrome:
            bits = ''.join(format(byte, '08b') for byte in page_data)
            pixel_values = [int(b) for b in bits]
            pixel_values = pixel_values + [0] * (bits_per_page - len(pixel_values))
        else:
            bits = ''.join(format(byte, '08b') for byte in page_data)
            pixel_values = []
            for i in range(0, len(bits) - 1, 2):
                pair = bits[i:i+2]
                pixel_values.append(int(pair, 2) if len(pair) == 2 else 0)
            pixel_values = pixel_values + [0] * ((bits_per_page // 2) - len(pixel_values))
        
        grid = []
        # Add 100 calibration rows with blocks of colors
        block_size = bits_per_row // (2 if monochrome else 4)
        calibration_row = []
        if monochrome:
            calibration_row += [0] * block_size  # White
            calibration_row += [1] * block_size  # Black
        else:
            calibration_row += [0] * block_size  # White
            calibration_row += [1] * block_size  # Red
            calibration_row += [2] * block_size  # Green
            calibration_row += [3] * (bits_per_row - 3 * block_size)  # Blue, adjust for division
        for _ in range(calibration_rows):
            grid.append(calibration_row[:])
        
        # Add data rows
        pixels_per_row = bits_per_row
        for row in range(bits_per_row - calibration_rows):
            start = row * pixels_per_row
            row_pixels = pixel_values[start:start + pixels_per_row]
            row_pixels = row_pixels + [0] * (pixels_per_row - len(row_pixels))
            grid.append(row_pixels)
        grids.append(grid)
    
    return grids, num_pages, filename

def create_bmp(grids, output_prefix, bits_per_row=2000, monochrome=False, dpi=300):
    color_map = {
        0: b'\xFF\xFF\xFF',  # White
        1: b'\xFF\x00\x00',  # Red
        2: b'\x00\xFF\x00',  # Green
        3: b'\x00\x00\xFF'   # Blue
    } if not monochrome else {
        0: b'\xFF\xFF\xFF',  # White
        1: b'\x00\x00\x00'   # Black
    }
    
    for page_num, grid in enumerate(grids):
        width = bits_per_row
        height = bits_per_row
        row_size = ((width * 24 + 31) // 32) * 4
        file_size = 54 + row_size * height
        bmp_data = bytearray()
        
        # BMP Header
        bmp_data.extend(b'BM')
        bmp_data.extend(struct.pack('<I', file_size))
        bmp_data.extend(struct.pack('<I', 0))
        bmp_data.extend(struct.pack('<I', 54))
        
        # DIB Header
        bmp_data.extend(struct.pack('<I', 40))
        bmp_data.extend(struct.pack('<i', width))
        bmp_data.extend(struct.pack('<i', -height))
        bmp_data.extend(struct.pack('<H', 1))
        bmp_data.extend(struct.pack('<H', 24))
        bmp_data.extend(struct.pack('<I', 0))
        bmp_data.extend(struct.pack('<I', 0))
        ppm = int(dpi * 39.37)
        bmp_data.extend(struct.pack('<i', ppm))
        bmp_data.extend(struct.pack('<i', ppm))
        bmp_data.extend(struct.pack('<I', 0))
        bmp_data.extend(struct.pack('<I', 0))
        
        # Pixel data
        for row in grid:
            for pixel_value in row:
                bmp_data.extend(color_map[pixel_value])
            padding = b'\x00' * (row_size - width * 3)
            bmp_data.extend(padding)
        
        output_file = f"{output_prefix}_{page_num + 1}.bmp"
        with open(output_file, 'wb') as f:
            f.write(bmp_data)
        print(f"BMP generated: {output_file} ({bits_per_row}x{bits_per_row} pixels)")

def decode_bmp(input_file, bits_per_row=2000, ec_bytes=0, monochrome=False):
    # Load image with Pillow
    img = Image.open(input_file)
    pixels = np.array(img)
    
    # Check dimensions
    height, width, _ = pixels.shape
    if width != bits_per_row or height != bits_per_row:
        raise ValueError(f"BMP dimensions must be {bits_per_row}x{bits_per_row}")
    
    # Define expected colors
    color_map = {
        0: np.array([255, 255, 255]),  # White
        1: np.array([255, 0, 0]),      # Red
        2: np.array([0, 255, 0]),      # Green
        3: np.array([0, 0, 255])       # Blue
    } if not monochrome else {
        0: np.array([255, 255, 255]),  # White
        1: np.array([0, 0, 0])         # Black
    }
    
    # Extract calibration blocks from first 100 rows
    calibration_rows = pixels[:100]
    block_size = bits_per_row // (2 if monochrome else 4)
    color_averages = []
    for i in range(2 if monochrome else 4):
        block_pixels = []
        for row in calibration_rows:
            block = row[i * block_size:(i + 1) * block_size]
            block_pixels.extend(block)
        avg_rgb = np.mean(block_pixels, axis=0)
        color_averages.append(avg_rgb)
    
    # Map data pixels (rows 100 to end) to closest calibrated color
    data_pixels = pixels[100:]
    pixel_values = []
    for row in data_pixels:
        for pixel in row:
            distances = [np.linalg.norm(pixel - avg) for avg in color_averages]
            closest = np.argmin(distances)
            pixel_values.append(closest)
    
    # Convert pixel values to bits
    if monochrome:
        bits = ''.join(str(p) for p in pixel_values)
    else:
        bits = ''.join(format(p, '02b') for p in pixel_values)
    
    # Convert bits to bytes
    byte_array = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) == 8:
            byte_array.append(int(byte_bits, 2))
    
    # Apply Reed-Solomon decoding
    if ec_bytes > 0:
        rs_n = 255
        rs_k = rs_n - ec_bytes
        rs = ReedSolomon(rs_n, rs_k)
        decoded_data = []
        for i in range(0, len(byte_array), rs_n):
            block = byte_array[i:i + rs_n]
            if len(block) == rs_n:
                decoded_data.extend(rs.decode(list(block)))
        byte_array = bytearray(decoded_data)
    
    return byte_array

def main():
    parser = argparse.ArgumentParser(description="Convert a file to A4-sized BMPs or decode from BMPs.")
    parser.add_argument("input_file", help="Input file to encode or BMP to decode")
    parser.add_argument("output_file", help="Output prefix for BMPs or file for decoded data")
    parser.add_argument("--bits-per-row", type=int, default=2000, help="Number of bits per row in the grid")
    parser.add_argument("--error-correction", type=int, default=0, help="Number of error correction bytes per block")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for BMP output")
    parser.add_argument("--monochrome", action="store_true", help="Use monochrome (black/white) instead of color")
    parser.add_argument("--decode", action="store_true", help="Decode from BMP instead of encoding")
    
    args = parser.parse_args()
    
    if args.bits_per_row <= 0:
        raise ValueError("bits-per-row must be positive")
    if args.error_correction < 0 or args.error_correction % 2 != 0:
        raise ValueError("error-correction must be non-negative and even")
    
    if args.decode:
        data = bytearray()
        page = 1
        base, _ = os.path.splitext(args.input_file)
        while True:
            input_file = args.input_file if page == 1 else f"{base}_{page}.bmp"
            if not os.path.exists(input_file):
                break
            decoded = decode_bmp(input_file, args.bits_per_row, args.error_correction, args.monochrome)
            data.extend(decoded)
            page += 1
        
        if not data:
            print("No data decoded")
            return
        
        filename_len = data[0]
        if filename_len >= len(data):
            print("Invalid filename length")
            return
        filename = data[1:1+filename_len].decode('utf-8', errors='ignore')
        data = data[1+filename_len:]
        
        with open(args.output_file, 'wb') as f:
            f.write(data)
        print(f"Data decoded and saved to {args.output_file} (Filename: {filename})")
    else:
        grids, num_pages, filename = file_to_bit_grid(args.input_file, args.bits_per_row, args.error_correction, args.monochrome)
        create_bmp(grids, os.path.splitext(args.output_file)[0], args.bits_per_row, args.monochrome, args.dpi)
        print(f"Generated {num_pages} BMP(s) for file {args.input_file}")

if __name__ == "__main__":
    main()
