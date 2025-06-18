import argparse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import math
import os
from reedsolo import RSCodec
from PIL import Image
import PyPDF2
import numpy as np

def file_to_bit_grid(filename, bits_per_row=300, ec_bytes=0):

    with open(filename, 'rb') as f:
        data = f.read()
    
    # Encode filename as bytes (prepend length to handle variable-length names)
    filename_bytes = os.path.basename(filename).encode('utf-8')
    filename_len = len(filename_bytes)
    if filename_len > 255:
        raise ValueError("Filename too long (max 255 bytes)")
    data = bytes([filename_len]) + filename_bytes + data
    
    # Convert bytes to bits
    bits = ''.join(format(byte, '08b') for byte in data)
    
    # Calculate grid dimensions
    bits_per_page = bits_per_row * bits_per_row
    if ec_bytes > 0:
        rs = RSCodec(ec_bytes)
        bytes_per_page = bits_per_page // 8 - ec_bytes
        if bytes_per_page <= 0:
            raise ValueError("Too many error correction bytes for grid size")
    else:
        bytes_per_page = bits_per_page // 8
    
    num_pages = math.ceil(len(data) / bytes_per_page)
    
    # Pad data with zeros if necessary
    data = data.ljust(num_pages * bytes_per_page, b'\x00')
    
    grids = []
    for page in range(num_pages):
        page_data = data[page * bytes_per_page:(page + 1) * bytes_per_page]
        if ec_bytes > 0:
            # Apply Reed-Solomon encoding
            page_data = rs.encode(page_data)
        page_bits = ''.join(format(byte, '08b') for byte in page_data)
        page_bits = page_bits.ljust(bits_per_page, '0')
        grid = []
        for row in range(bits_per_row):
            row_bits = page_bits[row * bits_per_row:(row + 1) * bits_per_row]
            grid.append([int(bit) for bit in row_bits])
        grids.append(grid)
    
    return grids, num_pages, filename

def create_pdf(grids, output_pdf, filename, bits_per_row=300, dot_size=2):
    c = canvas.Canvas(output_pdf, pagesize=letter)
    c.setAuthor(filename)
    c.setTitle(f"Data encoded from {filename}")
    page_width, page_height = letter
    margin = 50
    
    # Calculate scaling
    grid_size = bits_per_row * dot_size
    max_grid_width = page_width - 2 * margin
    max_grid_height = page_height - 2 * margin
    scale = min(max_grid_width / grid_size, max_grid_height / grid_size)
    
    scaled_dot_size = dot_size * scale
    scaled_grid_size = bits_per_row * scaled_dot_size
    
    x_offset = (page_width - scaled_grid_size) / 2
    y_offset = (page_height - scaled_grid_size) / 2
    
    for page_num, grid in enumerate(grids):
        c.setPageSize(letter)
        
        # Draw filename on first page
        if page_num == 0:
            c.setFont("Helvetica", 12)
            c.drawString(margin, page_height - margin, f"File: {filename}")
        
        # Draw grid
        for row in range(bits_per_row):
            for col in range(bits_per_row):
                if grid[row][col] == 1:
                    c.setFillColorRGB(0, 0, 0)
                    x = x_offset + col * scaled_dot_size
                    y = y_offset + (bits_per_row - row - 1) * scaled_dot_size
                    c.rect(x, y, scaled_dot_size, scaled_dot_size, stroke=0, fill=1)
        
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 12)
        c.drawString(margin, margin / 2, f"Page {page_num + 1}")
        
        c.showPage()
    
    c.save()
    print(f"PDF generated and saved at {output_pdf}")

def decode_image(image, bits_per_row=300, ec_bytes=0):
    # Convert to grayscale and threshold to binary
    img = image.convert('L')
    threshold = 128
    img = img.point(lambda p: 255 if p > threshold else 0)
    img = img.convert('1')  # Binary image
    
    # Resize to match grid size
    img = img.resize((bits_per_row, bits_per_row), Image.NEAREST)
    
    # Convert to bit grid
    pixels = np.array(img)
    grid = (pixels == 0).astype(int)  # Black (0) is 1, White (255) is 0
    
    # Flatten to bit string
    bits = ''.join(str(bit) for row in grid for bit in row)
    
    # Convert bits to bytes
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) == 8:
            byte_array.append(int(byte_bits, 2))
    
    # Apply error correction if enabled
    if ec_bytes > 0:
        rs = RSCodec(ec_bytes)
        try:
            byte_array = rs.decode(byte_array)[0]
        except Exception as e:
            print(f"Error correction failed: {e}")
            return None
    
    return byte_array

def decode_file(input_file, output_file, bits_per_row=300, ec_bytes=0):
    data = bytearray()
    
    if input_file.lower().endswith('.pdf'):
        with open(input_file, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf.pages)):
                page = pdf.pages[page_num]
                if page.get('/XObject'):
                    xobjects = page['/XObject'].get_object()
                    for obj in xobjects:
                        if xobjects[obj]['/Subtype'] == '/Image':
                            img = Image.open(xobjects[obj].get_stream())
                            decoded = decode_image(img, bits_per_row, ec_bytes)
                            if decoded:
                                data.extend(decoded)
    elif input_file.lower().endswith('.jpg'):
        img = Image.open(input_file)
        decoded = decode_image(img, bits_per_row, ec_bytes)
        if decoded:
            data.extend(decoded)
    else:
        raise ValueError("Input file must be a PDF or JPG")
    
    if not data:
        print("No data decoded")
        return
    
    # Extract filename
    filename_len = data[0]
    filename = data[1:1+filename_len].decode('utf-8', errors='ignore')
    data = data[1+filename_len:]
    
    # Save decoded data
    with open(output_file, 'wb') as f:
        f.write(data)
    
    print(f"Data decoded and saved to {output_file} (Filename: {filename})")

def main():
    parser = argparse.ArgumentParser(description="Convert a file to a printable PDF or decode from PDF/JPG.")
    parser.add_argument("input_file", help="Path to the input file (for encoding) or PDF/JPG (for decoding)")
    parser.add_argument("output_file", help="Path to save the output PDF or decoded file")
    parser.add_argument("--bits-per-row", type=int, default=300, help="Number of bits per row in the grid")
    parser.add_argument("--dot-size", type=float, default=2, help="Size of each bit square in points (encoding only)")
    parser.add_argument("--error-correction", type=int, default=0, help="Number of error correction bytes per page")
    parser.add_argument("--decode", action="store_true", help="Decode from PDF or JPG instead of encoding")
    
    args = parser.parse_args()
    
    if args.decode:
        decode_file(args.input_file, args.output_file, args.bits_per_row, args.error_correction)
    else:
        grids, num_pages, filename = file_to_bit_grid(args.input_file, args.bits_per_row, args.error_correction)
        create_pdf(grids, args.output_file, filename, args.bits_per_row, args.dot_size)
        print(f"Generated {num_pages} page(s) for file {args.input_file}")

if __name__ == "__main__":
    main()