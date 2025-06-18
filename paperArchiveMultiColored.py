import argparse
import math
import os
import struct

# Pure Python Reed–Solomon (no changes)
class ReedSolomon:
    def __init__(self, n, k):
        self.n, self.k = n, k
        self.poly = 0x11D
        self.exp = [0]*512
        self.log = [0]*256
        self._build_tables()
    def _build_tables(self):
        x = 1
        for i in range(512):
            self.exp[i] = x
            self.log[x] = i % 255
            x <<= 1
            if x & 0x100:
                x ^= self.poly
        self.log[0] = 0
    def _gf_mul(self, a, b):
        if a==0 or b==0: return 0
        return self.exp[(self.log[a]+self.log[b])%255]
    def _gf_div(self, a, b):
        if a==0: return 0
        if b==0: raise ValueError("Division by zero")
        return self.exp[(self.log[a]-self.log[b])%255]
    def _gf_poly_mul(self, p, q):
        r = [0]*(len(p)+len(q)-1)
        for j in range(len(q)):
            for i in range(len(p)):
                r[i+j] ^= self._gf_mul(p[i], q[j])
        return r
    def _gf_poly_div(self, num, den):
        quot = [0]*len(num)
        num = num[:]
        for i in range(len(num)-len(den)+1):
            coef = self._gf_div(num[i], den[0])
            quot[i] = coef
            for j in range(len(den)):
                num[i+j] ^= self._gf_mul(coef, den[j])
        return quot, num
    def encode(self, data):
        if len(data)!=self.k: raise ValueError(f"Expected {self.k}, got {len(data)}")
        gen = [1]
        for i in range(self.n-self.k):
            gen = self._gf_poly_mul(gen, [1, self.exp[i]])
        _, rem = self._gf_poly_div(data + [0]*(self.n-self.k), gen)
        return data + rem[-(self.n-self.k):]
    def decode(self, data):
        if len(data)!=self.n: raise ValueError(f"Expected {self.n}, got {len(data)}")
        gen = [1]
        for i in range(self.n-self.k):
            gen = self._gf_poly_mul(gen, [1, self.exp[i]])
        _, rem = self._gf_poly_div(data, gen)
        # no real correction; just return first k
        return data[:self.k]

# 3-bit color map
COLOR_MAP = {
    0: (0,0,0),   1: (255,0,0),   2: (0,255,0),   3: (0,0,255),
    4: (255,255,0),5: (255,0,255),6: (0,255,255),7: (255,255,255),
}

def file_to_bit_grid(filename, bits_per_row=2000, ec_bytes=0):
    raw = open(filename,'rb').read()
    name = os.path.basename(filename).encode('utf-8')
    if len(name)>255: raise ValueError("Filename too long")
    payload = bytes([len(name)]) + name + raw
    total_len = len(payload)
    payload = total_len.to_bytes(4,'big') + payload

    pixels_per_page = bits_per_row*bits_per_row
    bits_per_page   = pixels_per_page*3
    bytes_per_page  = bits_per_page//8 - ec_bytes if ec_bytes>0 else bits_per_page//8
    if bytes_per_page<=0: raise ValueError("Too many EC bytes")

    rs = ReedSolomon(bits_per_page//8, bytes_per_page) if ec_bytes>0 else None
    pages = math.ceil(len(payload)/bytes_per_page)
    padded = payload.ljust(pages*bytes_per_page, b'\x00')

    grids = []
    for p in range(pages):
        chunk = list(padded[p*bytes_per_page:(p+1)*bytes_per_page])
        if rs:
            chunk = rs.encode(chunk)
        bitstr = ''.join(f"{b:08b}" for b in chunk)

        # real bytes on this page (before padding)
        real_bytes = min(bytes_per_page, len(payload) - p*bytes_per_page)
        data_pixels = math.ceil((real_bytes + (ec_bytes or 0))*8/3)
        data_rows   = math.ceil(data_pixels/bits_per_row)
        total_bits  = data_rows*bits_per_row*3

        # pad bits to white
        pad = total_bits - len(bitstr)
        bitstr += '111'*math.ceil(pad/3)
        bitstr = bitstr[:total_bits]

        grid, idx = [], 0
        for _ in range(data_rows):
            row = []
            for _ in range(bits_per_row):
                trip = bitstr[idx:idx+3]
                if len(trip)<3: trip = trip.ljust(3,'1')
                row.append(int(trip,2))
                idx += 3
            grid.append(row)
        grids.append(grid)
    return grids

def create_bmp(grids, output_prefix, bits_per_row=2000, calibration_height=100):
    for i, grid in enumerate(grids, start=1):
        width     = bits_per_row
        data_rows = len(grid)
        height    = calibration_height + 1 + data_rows

        row_size  = ((width*24+31)//32)*4
        file_size = 54 + row_size*height

        bmp = bytearray()
        bmp.extend(b'BM')
        bmp.extend(struct.pack('<I', file_size))
        bmp.extend(b'\x00\x00\x00\x00')
        bmp.extend(struct.pack('<I', 54))

        bmp.extend(struct.pack('<I', 40))
        bmp.extend(struct.pack('<i', width))
        bmp.extend(struct.pack('<i', -height))  # top-down
        bmp.extend(struct.pack('<H', 1))
        bmp.extend(struct.pack('<H', 24))
        bmp.extend(b'\x00'*8)
        bmp.extend(struct.pack('<i',11811))
        bmp.extend(struct.pack('<i',11811))
        bmp.extend(b'\x00'*8)

        # calibration
        for _ in range(calibration_height):
            patch_w = width//8
            for x in range(width):
                idx = min(x//patch_w,7)
                r,g,b = COLOR_MAP[idx]
                bmp.extend(bytes((b,g,r)))
            bmp.extend(b'\x00'*(row_size-width*3))

        # white separator
        bmp.extend(b'\xFF\xFF\xFF'*width)
        bmp.extend(b'\x00'*(row_size-width*3))

        # data
        for row in grid:
            for idx in row:
                r,g,b = COLOR_MAP[idx]
                bmp.extend(bytes((b,g,r)))
            bmp.extend(b'\x00'*(row_size-width*3))

        out = f"{output_prefix}_{i}.bmp"
        with open(out,'wb') as f:
            f.write(bmp)
        print("Wrote", out)

def decode_bmp(input_file, bits_per_row=2000, ec_bytes=0, calibration_height=100):
    data = open(input_file,'rb').read()
    if data[:2]!=b'BM': raise ValueError("Not BMP")
    off   = struct.unpack('<I', data[10:14])[0]
    width = struct.unpack('<i', data[18:22])[0]
    h_raw = struct.unpack('<i', data[22:26])[0]
    height=abs(h_raw)
    bpp   = struct.unpack('<H', data[28:30])[0]
    if bpp!=24: raise ValueError("Only 24-bit")

    row_size = ((width*3+3)//4)*4
    bits=''

    for r in range(calibration_height+1, height):
        base = off + r*row_size
        line = data[base:base+row_size]
        for x in range(width):
            bb,gg,rr = line[x*3:x*3+3]
            best,dist = 7,1e9
            for k,(cr,cg,cb) in COLOR_MAP.items():
                d=(rr-cr)**2+(gg-cg)**2+(bb-cb)**2
                if d<dist: best,dist=k,d
            bits+=f"{best:03b}"

    ba=bytearray()
    for i in range(0,len(bits),8):
        bts=bits[i:i+8].ljust(8,'0')
        ba.append(int(bts,2))

    if ec_bytes:
        rs = ReedSolomon(len(ba), len(ba)-ec_bytes)
        ba = rs.decode(list(ba))

    L = int.from_bytes(ba[:4],'big')
    return ba[4:4+L]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_file")
    p.add_argument("output_file")
    p.add_argument("--bits-per-row", type=int, default=2000)
    p.add_argument("--error-correction", type=int, default=0)
    p.add_argument("--decode", action="store_true")
    args = p.parse_args()

    if args.decode:
        data = decode_bmp(args.input_file, args.bits_per_row, args.error_correction)
        name_len = data[0]
        name     = data[1:1+name_len].decode('utf-8',errors='ignore')
        body     = data[1+name_len:]
        with open(args.output_file,'wb') as f:
            f.write(body)
        print(f"Decoded {name} -> {args.output_file}")
    else:
        grids = file_to_bit_grid(args.input_file, args.bits_per_row, args.error_correction)
        create_bmp(grids, os.path.splitext(args.output_file)[0], args.bits_per_row)
        print(f"Generated {len(grids)} BMP(s)")

if __name__=="__main__":
    main()
