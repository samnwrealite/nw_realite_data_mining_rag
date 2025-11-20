# pipelines/run_ocr.py
import os
from ocr.extract_text import pdf_to_images, ocr_images

def run_ocr_for_pdf(pdf_path: str, images_dir="data/images", txt_out="data/ocr/output.txt"):
    print("Converting PDF to images...")
    image_paths = pdf_to_images(pdf_path, images_dir)
    print(f"Pages: {len(image_paths)}")
    print("Running OCR...")
    out = ocr_images(image_paths, txt_out)
    print("OCR complete:", out)
    return out

if __name__ == "__main__":
    # example: using uploaded file
    SAMPLE_PDF = "/mnt/data/APARTMENT NO G2 BLOCK G SIMBA VILLAS EMBAKASI NAIROBI COUNTY (1).pdf"
    run_ocr_for_pdf(SAMPLE_PDF, images_dir="data/images/sample", txt_out="data/ocr/sample_output.txt")
