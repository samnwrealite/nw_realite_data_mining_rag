# pipelines/run_all.py
import os
from pathlib import Path
from pipelines.run_ocr import pdf_to_images, ocr_images
from pipelines.run_embeddings import build_embeddings_for_textfile

POPPLER_PATH = r"C:\poppler\poppler-25.11.0\Library\bin"

PDF_DIR = r"C:\Users\samue\Documents\Work\Code\nw_realite_data_mining_rag\data\pdfs"
IMAGES_DIR = r"C:\Users\samue\Documents\Work\Code\nw_realite_data_mining_rag\data\images"
OCR_TXT = r"C:\Users\samue\Documents\Work\Code\nw_realite_data_mining_rag\data\ocr\output.txt"

def run_full(pdf_dir: str):
    Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(OCR_TXT)).mkdir(parents=True, exist_ok=True)

    all_texts = []
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"\nProcessing PDF: {pdf_file}")
            try:
                # Convert PDF to images
                pages = pdf_to_images(pdf_path, IMAGES_DIR, dpi=200, poppler_path=POPPLER_PATH)
                print(f"Converted {len(pages)} pages to images")
                # Run OCR
                txt_path = ocr_images(pages, OCR_TXT)
                all_texts.append(txt_path)
            except Exception as e:
                print(f"Failed to process {pdf_file}: {e}")

    # Build embeddings for the concatenated text
    for txt_file in all_texts:
        build_embeddings_for_textfile(txt_file)

    print("Pipeline finished.")

if __name__ == "__main__":
    run_full(PDF_DIR)
