import argparse
import math
import os
import struct
import numpy as np
from PIL import Image
from tqdm import tqdm

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
        if len(data) > self.k:
            raise ValueError(f"Data length {len(data)} exceeds k={self.k}")
        if len(data) < self.k:
            data = data + [0] * (self.k - len(data))
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
        print(f"Errors detected in data; correction not implemented")
        return data[:self.k]

def file_to_bit_grid(filename, bits_per_row=2000, ec_bytes=10, monochrome=False):
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
    calibration_rows = 100
    bits_per_row_pixels = bits_per_row * bits_per_pixel
    bytes_per_page = (bits_per_row_pixels * (bits_per_row - calibration_rows)) // 8
    
    rs_n = min(255, bytes_per_page)
    rs_k = rs_n - ec_bytes if ec_bytes > 0 else rs_n
    if ec_bytes > 0 and (rs_n <= rs_k or (rs_n - rs_k) % 2 != 0):
        raise ValueError(f"Invalid Reed-Solomon parameters: n={rs_n}, k={rs_k}")
    
    rs = ReedSolomon(rs_n, rs_k) if ec_bytes > 0 else None
    bytes_per_block = rs_k if rs else bytes_per_page
    num_blocks_per_page = math.ceil(bytes_per_page / bytes_per_block)
    bytes_per_page_effective = num_blocks_per_page * rs_n if rs else bytes_per_page
    
    num_full_pages = len(data) // bytes_per_page_effective
    remaining_bytes = len(data) % bytes_per_page_effective
    
    grids = []
    for page in tqdm(range(num_full_pages + (1 if remaining_bytes else 0)), desc="Encoding pages"):
        start = page * bytes_per_page_effective
        if page < num_full_pages:
            page_data = data[start:start + bytes_per_page_effective]
        else:
            page_data = data[start:start + remaining_bytes]
        
        if rs and page_data:
            encoded_data = []
            for i in range(0, len(page_data), rs_k):
                block = list(page_data[i:i + rs_k])
                if len(block) == rs_k:
                    encoded_data.extend(rs.encode(block))
                else:
                    encoded_data.extend(block)  # No padding
            page_data = bytes(encoded_data)
        
        if monochrome:
            bits = ''.join(format(byte, '08b') for byte in page_data)
            pixel_values = [int(b) for b in bits]
        else:
            bits = ''.join(format(byte, '08b') for byte in page_data)
            pixel_values = []
            for i in range(0, len(bits) - 1, 2):
                pair = bits[i:i+2]
                pixel_values.append(int(pair, 2) if len(pair) == 2 else 0)
        
        # Calculate exact rows for data
        pixels_per_row = bits_per_row
        total_pixels = len(pixel_values)
        data_rows = (total_pixels + pixels_per_row - 1) // pixels_per_row
        grid = []
        
        # Add 100 calibration rows
        block_size = bits_per_row // (2 if monochrome else 4)
        calibration_row = []
        if monochrome:
            calibration_row += [0] * block_size  # White
            calibration_row += [1] * block_size  # Black
        else:
            calibration_row += [0] * block_size  # White
            calibration_row += [1] * block_size  # Red
            calibration_row += [2] * block_size  # Green
            calibration_row += [3] * (bits_per_row - 3 * block_size)  # Blue
        for _ in range(calibration_rows):
            grid.append((calibration_row[:], bits_per_row))
        
        # Add data rows with exact pixel counts
        for row in range(data_rows):
            start = row * pixels_per_row
            end = min((row + 1) * pixels_per_row, total_pixels)
            row_pixels = pixel_values[start:end]
            row_width = end - start
            grid.append((row_pixels, row_width))
        
        # Add termination code (4 rows: Red, Green, Blue, White) with RS encoding
        if page == num_full_pages and remaining_bytes > 0:
            termination_data = [1] * rs_k + [2] * rs_k + [3] * rs_k + [0] * rs_k  # RGBW pattern
            encoded_termination = []
            for i in range(0, len(termination_data), rs_k):
                block = termination_data[i:i + rs_k]
                if len(block) == rs_k:
                    encoded_termination.extend(rs.encode(block))
            termination_pixels = []
            for byte in encoded_termination:
                bits = format(byte, '08b')
                for i in range(0, len(bits) - 1, 2):
                    pair = bits[i:i+2]
                    termination_pixels.append(int(pair, 2) if len(pair) == 2 else 0)
            termination_rows = (len(termination_pixels) + bits_per_row - 1) // bits_per_row
            for row in range(termination_rows):
                start = row * bits_per_row
                end = min((row + 1) * bits_per_row, len(termination_pixels))
                row_pixels = termination_pixels[start:end]
                row_width = end - start
                grid.append((row_pixels, row_width))
        
        grids.append((grid, calibration_rows + data_rows + (termination_rows if page == num_full_pages and remaining_bytes > 0 else 0)))
    
    return grids, len(grids), filename

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
    
    for page_num, (grid, height) in enumerate(tqdm(grids, desc="Creating BMPs")):
        bmp_data = bytearray()
        file_size = 54
        
        # Calculate file size based on actual row widths
        for row_pixels, row_width in grid:
            row_size = ((row_width * 24 + 31) // 32) * 4
            file_size += row_size
        
        # BMP Header
        bmp_data.extend(b'BM')
        bmp_data.extend(struct.pack('<I', file_size))
        bmp_data.extend(struct.pack('<I', 0))
        bmp_data.extend(struct.pack('<I', 54))
        
        # DIB Header
        bmp_data.extend(struct.pack('<I', 40))
        bmp_data.extend(struct.pack('<i', bits_per_row))  # Max width
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
        
        # Pixel data with exact row widths
        for row_pixels, row_width in grid:
            for pixel_value in row_pixels:
                bmp_data.extend(color_map[pixel_value])
            row_size = ((row_width * 24 + 31) // 32) * 4
            padding = b'\x00' * (row_size - row_width * 3)
            bmp_data.extend(padding)
        
        output_file = f"{output_prefix}_{page_num + 1}.bmp"
        with open(output_file, 'wb') as f:
            f.write(bmp_data)
        print(f"BMP generated: {output_file} ({bits_per_row}x{height} pixels, last row {grid[-1][1]} pixels)")

def decode_bmp(input_file, bits_per_row=2000, ec_bytes=10, monochrome=False):
    print(f"Attempting to decode {input_file}")
    try:
        img = Image.open(input_file)
        pixels = np.array(img)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return bytearray()
    
    height, width, _ = pixels.shape
    if width != bits_per_row or height < 100:
        print(f"Error: BMP width must be {bits_per_row} and height at least 100, got {width}x{height}")
        return bytearray()
    
    color_map = {
        0: np.array([255, 255, 255]),  # White
        1: np.array([255, 0, 0]),      # Red
        2: np.array([0, 255, 0]),      # Green
        3: np.array([0, 0, 255])       # Blue
    } if not monochrome else {
        0: np.array([255, 255, 255]),  # White
        1: np.array([0, 0, 0])         # Black
    }
    
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
    
    data_pixels = pixels[100:height]
    pixel_values = []
    termination_detected = False
    for row_idx, row in enumerate(tqdm(data_pixels, desc=f"Decoding pixels from {os.path.basename(input_file)}")):
        pixels_to_read = min(len(row), bits_per_row)
        row_values = []
        for pixel in row[:pixels_to_read]:
            distances = [np.linalg.norm(pixel - avg) for avg in color_averages]
            closest = np.argmin(distances)
            row_values.append(closest)
        
        # Check for termination code (RGBW sequence over 4 rows)
        if row_idx >= len(data_pixels) - 4 and not termination_detected:
            last_four = data_pixels[row_idx-3:row_idx+1] if row_idx >= 3 else data_pixels[:row_idx+1] + [row] * (4 - row_idx)
            if all(len(r) >= block_size for r in last_four):
                patterns = []
                for r in last_four:
                    block = r[:block_size]
                    avg_rgb = np.mean(block, axis=0)
                    distances = [np.linalg.norm(avg_rgb - avg) for avg in color_averages]
                    patterns.append(np.argmin(distances))
                if patterns == [1, 2, 3, 0]:  # RGBW
                    termination_detected = True
                    break
        
        pixel_values.extend(row_values[:bits_per_row])
    
    if monochrome:
        bits = ''.join(str(p) for p in pixel_values)
    else:
        bits = ''.join(format(p, '02b') for p in pixel_values)
    
    byte_array = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) == 8:
            byte_array.append(int(byte_bits, 2))
    
    if ec_bytes > 0:
        rs_n = 255
        rs_k = rs_n - ec_bytes
        rs = ReedSolomon(rs_n, rs_k)
        decoded_data = []
        for i in range(0, len(byte_array), rs_n):
            block = byte_array[i:i + rs_n]
            if len(block) == rs_n:
                decoded_data.extend(rs.decode(list(block)))
            else:
                print(f"Processing partial block in {input_file}, length {len(block)}")
                decoded_data.extend(block[:rs_k])
        byte_array = bytearray(decoded_data)
    
    print(f"Decoded {len(byte_array)} bytes from {input_file}")
    return byte_array

def main():
    parser = argparse.ArgumentParser(description="Convert a file to A4-sized BMPs or decode from BMPs.")
    parser.add_argument("input_file", help="Input file to encode or BMP to decode")
    parser.add_argument("output_file", help="Output prefix for BMPs or file for decoded data")
    parser.add_argument("--bits-per-row", type=int, default=2000, help="Number of bits per row in the grid")
    parser.add_argument("--error-correction", type=int, default=10, help="Number of error correction bytes per block")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for BMP output")
    parser.add_argument("--monochrome", action="store_true", help="Use monochrome (black/white) instead of color")
    parser.add_argument("--decode", action="store_true", help="Decode from BMP instead of encoding")
    
    args = parser.parse_args()
    
    if args.bits_per_row <= 0:
        raise ValueError("bits-per-row must be positive")
    if args.error_correction < 0 or args.error_correction % 2 != 0:
        raise ValueError("error-correction must be non-negative and even")
    
    if args.decode:
        input_file_abs = os.path.abspath(args.input_file)
        base, ext = os.path.splitext(input_file_abs)
        if base.endswith('_1'):
            base = base[:-2]
        print(f"Base path for BMPs: {base}")
        found_bmps = []
        page = 1
        while True:
            check_file = input_file_abs if page == 1 else f"{base}_{page}.bmp"
            print(f"Checking for BMP: {check_file}")
            if os.path.exists(check_file):
                found_bmps.append(check_file)
            else:
                break
            page += 1
        if not found_bmps:
            print(f"No BMP files found starting with {input_file_abs}")
            return
        print("Found BMP files:")
        for bmp in found_bmps:
            print(f" - {bmp}")
        
        data = bytearray()
        with tqdm(total=len(found_bmps), desc="Decoding BMPs") as pbar:
            for page, input_file in enumerate(found_bmps, 1):
                decoded = decode_bmp(input_file, args.bits_per_row, args.error_correction, args.monochrome)
                data.extend(decoded)
                print(f"Total data size after decoding {input_file}: {len(data)} bytes")
                pbar.update(1)
        
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
        create_bmp(grids, os.path.splitext(os.path.abspath(args.output_file))[0], args.bits_per_row, args.monochrome, args.dpi)
        print(f"Generated {num_pages} BMP(s) for file {args.input_file}")

if __name__ == "__main__":
    main()
