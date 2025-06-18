# PaperArchive

A set of Python command-line tools to encode arbitrary files into printable bit-grid images or PDFs, and decode them back.  Use your printer (or any viewer) and a simple scanner or camera to archive and recover data on paper.

---

## Features

- **Black-and-White BMP** (`paperArchiveBMP.py`)  
  — Encode any file into one or more A4-sized 24-bit BMPs (2000×2000 grid), decode back to the original file.  
- **Multi-Color BMP** (`paperArchiveMultiColored.py`)  
  — Same as above, but encodes 3 bits per pixel in 8 colors, includes an 8-patch calibration strip for reliable scanning.  
- **Printable PDF** (`paperArchive.py`)  
  — Packs the bit-grid into letter-size pages (PDF), with adjustable dot size, optional Reed–Solomon error correction, and ability to decode from PDF or JPG.  
- **Error Correction**  
  — Optional Reed–Solomon parity bytes per page to recover from printing/scanning defects.  
- **Filename Embedding**  
  — Automatically stores and restores the original filename.  
- **Simple Decoding**  
  — Built-in `--decode` mode reads back images (BMP/PDF/JPG) and reconstructs the original file.

---

## Requirements

- Python 3.7+  
- [Pillow](https://pypi.org/project/Pillow/)  
- [reedsolo](https://pypi.org/project/reedsolo/) (for `paperArchive.py` PDF/JPG)  
- [reportlab](https://pypi.org/project/reportlab/) (for PDF generation)  
- [PyPDF2](https://pypi.org/project/PyPDF2/) (for PDF decoding)  
- Standard library modules: `argparse`, `math`, `os`, `struct`

Install with:
```bash
pip install Pillow reedsolo reportlab PyPDF2
