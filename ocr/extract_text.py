# ocr/extract_text.py
import os
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from PIL import Image
from pathlib import Path
from tqdm import tqdm

ocr = PaddleOCR(use_angle_cls=True, lang="en")  # initialize once

def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 200, poppler_path: str = None):
    """
    Convert a PDF to images.
    - pdf_path: path to PDF file
    - out_dir: directory to save images
    - dpi: resolution for images
    - poppler_path: required on Windows if Poppler is not on PATH
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    paths = []
    for i, page in enumerate(pages, start=1):
        out_path = os.path.join(out_dir, f"{Path(pdf_path).stem}_page_{i:04d}.png")
        page.save(out_path, "PNG")
        paths.append(out_path)
    return paths

def ocr_images(image_paths, out_txt_path):
    """
    Run PaddleOCR on a list of images and save output text.
    """
    Path(os.path.dirname(out_txt_path)).mkdir(parents=True, exist_ok=True)
    all_text = []
    for img_path in tqdm(image_paths, desc="OCR pages"):
        result = ocr.ocr(img_path)
        page_lines = []
        for line in result:
            try:
                page_lines.append(line[1][0])
            except Exception:
                continue
        page_text = "\n".join(page_lines)
        all_text.append(page_text)
    full_text = "\n\n".join(all_text)
    with open(out_txt_path, "w", encoding="utf8") as f:
        f.write(full_text)
    return out_txt_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdfs_dir", default="data/pdfs", help="Directory containing PDF files")
    parser.add_argument("--out_dir", default="data/images", help="Directory to save images")
    parser.add_argument("--txt_out", default="data/ocr/output.txt", help="Output text file path")
    parser.add_argument("--poppler_path", default=r"C:\poppler-23.10.0\bin", help="Poppler bin path for Windows")
    args = parser.parse_args()

    all_texts = []
    for pdf_file in os.listdir(args.pdfs_dir):
        if pdf_file.lower().endswith(".pdf"):
            pdf_path = os.path.join(args.pdfs_dir, pdf_file)
            print(f"\nProcessing PDF: {pdf_file}")
            try:
                images = pdf_to_images(pdf_path, args.out_dir, dpi=200, poppler_path=args.poppler_path)
                print(f"Saved {len(images)} images from {pdf_file}")
                txt_path = ocr_images(images, args.txt_out)
                all_texts.append(txt_path)
            except Exception as e:
                print(f"Failed to process {pdf_file}: {e}")

    print(f"OCR complete for {len(all_texts)} PDFs. Text saved to {args.txt_out}")
