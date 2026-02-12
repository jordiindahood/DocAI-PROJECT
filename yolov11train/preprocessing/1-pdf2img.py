#!/usr/bin/env python3
"""
Step 1: PDF -> PNG converter (chunked) for YOLO training.
Reads settings from data.yaml.
"""

from pdf2image import convert_from_path, pdfinfo_from_path
from pathlib import Path
import tempfile
import yaml

# Load config from data.yaml
CONFIG_PATH = Path(__file__).resolve().parent.parent / "data.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PDF_DIR = PROJECT_ROOT / cfg["paths"]["pdf_dir"]
IMG_DIR = PROJECT_ROOT / cfg["paths"]["image_dir"]

pdf2img = cfg["pdf2img"]
DPI = pdf2img["dpi"]
CHUNK_PAGES = pdf2img["chunk_pages"]
MAX_PAGES_PER_PDF = pdf2img["max_pages_per_pdf"]
THREAD_COUNT = pdf2img["thread_count"]
FMT = "png"

if not PDF_DIR.exists():
    raise SystemExit(f"Error: PDF directory not found at {PDF_DIR}")

IMG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Converting PDFs from {PDF_DIR} to images...")
print(f"Output directory: {IMG_DIR}")
print(f"DPI: {DPI} | chunk: {CHUNK_PAGES} pages | threads: {THREAD_COUNT}")
print("-" * 60)

total_pages = 0

with tempfile.TemporaryDirectory(prefix="pdf2img_") as tmpdir:
    tmpdir = Path(tmpdir)

    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        try:
            info = pdfinfo_from_path(str(pdf))
            n_pages = int(info.get("Pages", 0))
            if n_pages <= 0:
                print(f"  x {pdf.name}: couldn't read page count")
                continue

            if MAX_PAGES_PER_PDF and n_pages > MAX_PAGES_PER_PDF:
                n_pages = MAX_PAGES_PER_PDF

            print(f"Processing {pdf.name} ({n_pages} pages)...")

            start = 1
            while start <= n_pages:
                end = min(start + CHUNK_PAGES - 1, n_pages)

                try:
                    pages = convert_from_path(
                        str(pdf),
                        dpi=DPI,
                        first_page=start,
                        last_page=end,
                        fmt=FMT,
                        thread_count=THREAD_COUNT,
                        output_folder=str(tmpdir),
                        paths_only=False,
                        use_pdftocairo=True,
                    )

                    for i, page in enumerate(pages, start=start):
                        out = IMG_DIR / f"{pdf.stem}_page{i:04d}.png"
                        page.save(out, "PNG")
                        total_pages += 1

                    print(f"  pages {start}-{end}")

                except Exception as e:
                    print(f"  x {pdf.name} pages {start}-{end}: {e}")

                start = end + 1

        except Exception as e:
            print(f"  x Error processing {pdf.name}: {e}")

print("-" * 60)
print(f"Conversion complete. Total pages: {total_pages}")
