import argparse
import math
import os
import struct

# Pure Python Reed-Solomon implementation
class ReedSolomon:
    def __init__(self, n, k):
        self.n = n  # Total bytes (data + parity)
        self.k = k  # Data bytes
        self.t = (n - k) // 2  # Error correction capability
        self.poly = 0x11D  # GF(2^8) polynomial: x^8 + x^4 + x^3 + x^2 + 1
        self.exp = [0] * 512
        self.log = [0] * 256
        self._build_tables()

    def _build_tables(self):
        x = 1
        for i in range(512):
            self.exp[i] = x
            self.log[x] = i % 255
            x <<= 1
            if x & 0x100:
                x ^= self.poly
        self.log[0] = 0  # Log(0) is undefined, but set to 0 for convenience

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
        quot = [0] * len(num)
        num = num[:]
        for i in range(len(num) - len(den) + 1):
            coef = self._gf_div(num[i], den[0])
            quot[i] = coef
            for j in range(len(den)):
                num[i + j] ^= self._gf_mul(coef, den[j])
        return quot, num

    def encode(self, data):
        if len(data) != self.k:
            raise ValueError(f"Expected {self.k} data bytes, got {len(data)}")
        gen = [1]
        for i in range(self.n - self.k):
            gen = self._gf_poly_mul(gen, [1, self.exp[i]])
        _, rem = self._gf_poly_div(data + [0] * (self.n - self.k), gen)
        return data + rem[-(self.n - self.k):]

    def decode(self, data):
        if len(data) != self.n:
            raise ValueError(f"Expected {self.n} bytes, got {len(data)}")
        try:
            gen = [1]
            for i in range(self.n - self.k):
                gen = self._gf_poly_mul(gen, [1, self.exp[i]])
            _, rem = self._gf_poly_div(data, gen)
            if all(r == 0 for r in rem):
                return data[:self.k]
            else:
                print("Errors detected; correction not implemented")
                return data[:self.k]
        except Exception:
            return data[:self.k]

def file_to_bit_grid(filename, bits_per_row=2000, ec_bytes=0):
    with open(filename, 'rb') as f:
        data = f.read()
    
    # Encode filename
    filename_bytes = os.path.basename(filename).encode('utf-8')
    filename_len = len(filename_bytes)
    if filename_len > 255:
        raise ValueError("Filename too long (max 255 bytes)")
    data = bytes([filename_len]) + filename_bytes + data
    
    # Calculate grid dimensions
    bits_per_page = bits_per_row * bits_per_row
    if ec_bytes > 0:
        bytes_per_page = bits_per_page // 8 - ec_bytes
        if bytes_per_page <= 0:
            raise ValueError("Too many error correction bytes for grid size")
        rs = ReedSolomon(bits_per_page // 8, bytes_per_page)
    else:
        bytes_per_page = bits_per_page // 8
        rs = None
    
    num_pages = math.ceil(len(data) / bytes_per_page)
    data = data.ljust(num_pages * bytes_per_page, b'\x00')
    
    grids = []
    for page in range(num_pages):
        page_data = data[page * bytes_per_page:(page + 1) * bytes_per_page]
        if rs:
            page_data = rs.encode(list(page_data))
        page_bits = ''.join(format(byte, '08b') for byte in page_data)
        page_bits = page_bits.ljust(bits_per_page, '0')
        grid = []
        for row in range(bits_per_row):
            row_bits = page_bits[row * bits_per_row:(row + 1) * bits_per_row]
            grid.append([int(bit) for bit in row_bits])
        grids.append(grid)
    
    return grids, num_pages, filename

def create_bmp(grids, output_prefix, bits_per_row=2000):
    for page_num, grid in enumerate(grids):
        width = bits_per_row
        height = bits_per_row
        # BMP header (14 bytes) + DIB header (40 bytes, BITMAPINFOHEADER)
        row_size = ((width * 24 + 31) // 32) * 4  # Padded to 4 bytes
        file_size = 54 + row_size * height
        bmp_data = bytearray()
        
        # BMP Header
        bmp_data.extend(b'BM')  # Signature
        bmp_data.extend(struct.pack('<I', file_size))  # File size
        bmp_data.extend(struct.pack('<I', 0))  # Reserved
        bmp_data.extend(struct.pack('<I', 54))  # Pixel data offset
        
        # DIB Header
        bmp_data.extend(struct.pack('<I', 40))  # DIB header size
        bmp_data.extend(struct.pack('<i', width))  # Width
        # Use negative height for bottom-up BMP to simplify printing
        bmp_data.extend(struct.pack('<i', -height))  # Height (negative for bottom-up)
        bmp_data.extend(struct.pack('<H', 1))  # Planes
        bmp_data.extend(struct.pack('<H', 24))  # Bits per pixel
        bmp_data.extend(struct.pack('<I', 0))  # Compression (none)
        bmp_data.extend(struct.pack('<I', 0))  # Image size (0 for uncompressed)
        bmp_data.extend(struct.pack('<i', 11811))  # X pixels per meter (~300 DPI)
        bmp_data.extend(struct.pack('<i', 11811))  # Y pixels per meter
        bmp_data.extend(struct.pack('<I', 0))  # Colors used
        bmp_data.extend(struct.pack('<I', 0))  # Important colors
        
        # Pixel data (bottom-up, BGR format)
        for row in range(height):  # Bottom-up due to negative height
            for col in range(width):
                bit = grid[row][col]
                # Black (1) = (0,0,0), White (0) = (255,255,255)
                pixel = b'\x00\x00\x00' if bit == 1 else b'\xFF\xFF\xFF'
                bmp_data.extend(pixel)
            padding = b'\x00' * (row_size - width * 3)
            bmp_data.extend(padding)
        
        output_file = f"{output_prefix}_{page_num + 1}.bmp"
        with open(output_file, 'wb') as f:
            f.write(bmp_data)
        print(f"BMP generated: {output_file} ({bits_per_row}x{bits_per_row} pixels)")

def decode_bmp(input_file, bits_per_row=2000, ec_bytes=0):
    with open(input_file, 'rb') as f:
        data = f.read()
    
    # Parse BMP header
    if data[:2] != b'BM':
        raise ValueError("Not a BMP file")
    offset = struct.unpack('<I', data[10:14])[0]
    width = struct.unpack('<i', data[18:22])[0]
    height = struct.unpack('<i', data[22:26])[0]
    bpp = struct.unpack('<H', data[28:30])[0]
    if bpp != 24:
        raise ValueError("Only 24-bit BMP supported")
    if abs(width) != bits_per_row or abs(height) != bits_per_row:
        raise ValueError(f"BMP dimensions must be {bits_per_row}x{bits_per_row}")
    
    # Handle bottom-up (negative height) or top-down
    is_bottom_up = height < 0
    height = abs(height)
    row_size = ((width * 3 + 3) // 4) * 4
    grid = []
    for row in range(height):
        # Adjust row index for bottom-up BMPs
        row_idx = row if is_bottom_up else (height - 1 - row)
        row_data = data[offset + row_idx * row_size:offset + (row_idx + 1) * row_size]
        row_bits = []
        for col in range(width):
            pixel = row_data[col * 3:col * 3 + 3]
            bit = 1 if pixel == b'\x00\x00\x00' else 0
            row_bits.append(bit)
        grid.append(row_bits)
    
    # Flatten to bit string
    bits = ''.join(str(bit) for row in grid for bit in row)
    
    # Convert to bytes
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) == 8:
            byte_array.append(int(byte_bits, 2))
    
    # Apply error correction
    if ec_bytes > 0:
        rs = ReedSolomon(len(byte_array), len(byte_array) - ec_bytes)
        byte_array = rs.decode(byte_array)
    
    return byte_array

def main():
    parser = argparse.ArgumentParser(description="Convert a file to A4-sized BMPs or decode from BMPs.")
    parser.add_argument("input_file", help="Input file to encode or BMP to decode")
    parser.add_argument("output_file", help="Output prefix for BMPs or file for decoded data")
    parser.add_argument("--bits-per-row", type=int, default=2000, help="Number of bits per row in the grid (A4-sized)")
    parser.add_argument("--error-correction", type=int, default=0, help="Number of error correction bytes per page")
    parser.add_argument("--decode", action="store_true", help="Decode from BMP instead of encoding")
    
    args = parser.parse_args()
    
    if args.decode:
        data = bytearray()
        page = 1
        base, _ = os.path.splitext(args.input_file)
        while True:
            input_file = args.input_file if page == 1 else f"{base}_{page}.bmp"
            if not os.path.exists(input_file):
                break
            decoded = decode_bmp(input_file, args.bits_per_row, args.error_correction)
            if decoded:
                data.extend(decoded)
            page += 1
        
        if not data:
            print("No data decoded")
            return
        
        # Extract filename
        filename_len = data[0]
        filename = data[1:1+filename_len].decode('utf-8', errors='ignore')
        data = data[1+filename_len:]
        
        with open(args.output_file, 'wb') as f:
            f.write(data)
        print(f"Data decoded and saved to {args.output_file} (Filename: {filename})")
    else:
        grids, num_pages, filename = file_to_bit_grid(args.input_file, args.bits_per_row, args.error_correction)
        create_bmp(grids, os.path.splitext(args.output_file)[0], args.bits_per_row)
        print(f"Generated {num_pages} BMP(s) for file {args.input_file}")

if __name__ == "__main__":
    main()