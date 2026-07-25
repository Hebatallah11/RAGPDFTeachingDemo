import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain_core.documents import Document

# Bump this up if OCR text looks blurry/garbled on your scans
OCR_DPI = 300
# If a "real" text page has fewer than this many characters, treat it as scanned
MIN_TEXT_LEN_THRESHOLD = 20


def load_pdf_with_ocr_fallback(file_path: str) -> list[Document]:
    """
    Drop-in replacement for PyPDFLoader().load().
    For each page: try native text extraction first; if the page has
    little/no extractable text (i.e. it's a scanned image), rasterize
    the page and run OCR instead.
    Returns Document objects with the same metadata shape PyPDFLoader uses
    (source, page), so rag_service.py needs no changes downstream.
    """
    documents = []
    pdf = fitz.open(file_path)

    for page_number in range(len(pdf)):
        page = pdf[page_number]
        native_text = page.get_text().strip()

        if len(native_text) >= MIN_TEXT_LEN_THRESHOLD:
            page_text = native_text
        else:
            page_text = _ocr_page(page)

        documents.append(
            Document(
                page_content=page_text,
                metadata={"source": file_path, "page": page_number},
            )
        )

    pdf.close()
    return documents


def _ocr_page(page: "fitz.Page") -> str:
    zoom = OCR_DPI / 72  # fitz's default page resolution is 72 dpi
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(image).strip()
