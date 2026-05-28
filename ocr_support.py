# =========================
# OCR
# =========================

import os
import pickle
import faiss
import fitz
import cv2
import pytesseract
import numpy as np

from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =========================
# FOLDERS
# =========================

DOCUMENTS_FOLDER = "sample"

# =========================
# LOAD MODEL
# =========================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)

print("Model loaded!")

# =========================
# OCR PDF TEXT EXTRACTOR
# =========================

def extract_text_from_pdf(pdf_path):

    full_text = ""

    try:

        doc = fitz.open(pdf_path)

        print(f"Reading: {pdf_path}")

        for page_num in range(len(doc)):

            page = doc[page_num]

            # =========================
            # NORMAL TEXT EXTRACTION
            # =========================

            text = page.get_text()

            # If normal PDF text exists
            if text.strip():

                full_text += text + " "

            else:

                print(
                    f"OCR Page: {page_num + 1}"
                )

                # =========================
                # CONVERT PAGE TO IMAGE
                # =========================

                pix = page.get_pixmap()

                img = Image.frombytes(

                    "RGB",

                    [pix.width, pix.height],

                    pix.samples
                )

                img_np = np.array(img)

                # =========================
                # IMAGE PREPROCESSING
                # =========================

                gray = cv2.cvtColor(

                    img_np,

                    cv2.COLOR_RGB2GRAY
                )

                # Noise removal
                gray = cv2.GaussianBlur(

                    gray,

                    (3, 3),

                    0
                )

                # Thresholding
                gray = cv2.threshold(

                    gray,

                    0,

                    255,

                    cv2.THRESH_BINARY + cv2.THRESH_OTSU

                )[1]

                # =========================
                # OCR
                # =========================

                ocr_text = pytesseract.image_to_string(
                    gray
                )

                full_text += ocr_text + " "

        doc.close()

    except Exception as e:

        print(
            "ERROR:",
            pdf_path,
            e
        )

    return full_text

# =========================
# SPLIT TEXT INTO CHUNKS
# =========================

def split_text(text, chunk_size=500):

    words = text.split()

    chunks = []

    for i in range(

        0,

        len(words),

        chunk_size
    ):

        chunk = " ".join(

            words[i:i + chunk_size]

        )

        chunks.append(chunk)

    return chunks

# =========================
# READ DOCUMENTS
# =========================

all_chunks = []

metadata = []

print("Scanning documents...")

for filename in tqdm(

    os.listdir(DOCUMENTS_FOLDER)

):

    if not filename.lower().endswith(".pdf"):

        continue

    pdf_path = os.path.join(

        DOCUMENTS_FOLDER,

        filename
    )

    # =========================
    # EXTRACT TEXT
    # =========================

    text = extract_text_from_pdf(
        pdf_path
    )

    if not text.strip():

        print(
            f"No text found: {filename}"
        )

        continue

    # =========================
    # SPLIT INTO CHUNKS
    # =========================

    chunks = split_text(
        text,
        chunk_size=500
    )

    for chunk in chunks:

        if len(chunk.strip()) < 20:

            continue

        all_chunks.append(chunk)

        metadata.append({

            "file": filename,

            "text": chunk

        })

print(
    f"Total Chunks: {len(all_chunks)}"
)

# =========================
# CREATE EMBEDDINGS
# =========================

print("Creating embeddings...")

embeddings = model.encode(

    all_chunks,

    batch_size=32,

    show_progress_bar=True,

    convert_to_numpy=True
)

embeddings = embeddings.astype(
    "float32"
)

# =========================
# NORMALIZE EMBEDDINGS
# =========================

faiss.normalize_L2(
    embeddings
)

# =========================
# CREATE FAISS INDEX
# =========================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)

print(
    f"Indexed Vectors: {index.ntotal}"
)

# =========================
# SAVE INDEX
# =========================

faiss.write_index(

    index,

    "document_index.faiss"
)

# =========================
# SAVE METADATA
# =========================

with open(

    "metadata.pkl",

    "wb"

) as f:

    pickle.dump(
        metadata,
        f
    )

print("Index saved successfully!")
