'''# __define-ocg__

import os
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

# =========================
# Load Embedding Model
# =========================

print("Loading model...")

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

print("Model loaded!")

# =========================
# Document Folder
# =========================

DOCUMENT_FOLDER = "sample"

# =========================
# Read PDF
# =========================

def read_pdf(path):

    text = ""

    reader = PdfReader(path)

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:
            text += extracted + " "

    return text

# =========================
# Read DOCX
# =========================

def read_docx(path):

    doc = Document(path)

    return "\n".join(
        [p.text for p in doc.paragraphs]
    )

# =========================
# Read TXT
# =========================

def read_txt(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()

# =========================
# Chunk Text
# =========================

def chunk_text(
    text,
    chunk_size=300
):

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
# Process Documents
# =========================

all_embeddings = []

print("Processing documents...")

for file in os.listdir(DOCUMENT_FOLDER):

    path = os.path.join(
        DOCUMENT_FOLDER,
        file
    )

    content = ""

    # PDF
    if file.endswith(".pdf"):

        content = read_pdf(path)

    # DOCX
    elif file.endswith(".docx"):

        content = read_docx(path)

    # TXT
    elif file.endswith(".txt"):

        content = read_txt(path)

    else:
        continue

    print(f"\nProcessing: {file}")

    # Split into chunks
    chunks = chunk_text(content)

    print(f"Chunks created: {len(chunks)}")

    # Create embeddings
    for chunk in chunks:

        embedding = model.encode(chunk)

        all_embeddings.append({

            "file": file,
            "text": chunk,
            "embedding": embedding

        })

    print("Embeddings created!")

print("\nDONE!")
print(f"Total embeddings: {len(all_embeddings)}")'''






# __define-ocg__ FULL EMBEDDING + FAISS INDEX CODE

import os
import pickle
import numpy as np
import faiss

from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer

# =========================
# Load Embedding Model
# =========================

print("Loading model...")

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

print("Model loaded!")

# =========================
# Document Folder
# =========================

DOCUMENT_FOLDER = "sample"

# =========================
# Store All Data
# =========================

all_embeddings = []

# =========================
# Read PDF Function
# =========================

def read_pdf(path):

    text = ""

    reader = PdfReader(path)

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:

            text += extracted + " "

    return text

# =========================
# Read DOCX Function
# =========================

def read_docx(path):

    doc = Document(path)

    return "\n".join(

        [p.text for p in doc.paragraphs]

    )

# =========================
# Read TXT Function
# =========================

def read_txt(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()

# =========================
# Chunk Text Function
# =========================

def chunk_text(
    text,
    chunk_size=300
):

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
# Process Documents
# =========================

print("\nProcessing documents...")

for file in os.listdir(DOCUMENT_FOLDER):

    path = os.path.join(
        DOCUMENT_FOLDER,
        file
    )

    content = ""

    # =========================
    # PDF
    # =========================

    if file.endswith(".pdf"):

        content = read_pdf(path)

    # =========================
    # DOCX
    # =========================

    elif file.endswith(".docx"):

        content = read_docx(path)

    # =========================
    # TXT
    # =========================

    elif file.endswith(".txt"):

        content = read_txt(path)

    else:

        continue

    print(f"\nProcessing: {file}")

    # =========================
    # Create Chunks
    # =========================

    chunks = chunk_text(content)

    print(f"Chunks created: {len(chunks)}")

    # =========================
    # Generate Embeddings
    # =========================

    for chunk in chunks:

        embedding = model.encode(chunk)

        all_embeddings.append({

            "file": file,
            "text": chunk,
            "embedding": embedding

        })

    print("Embeddings created!")

# =========================
# Convert Embeddings to NumPy
# =========================

embedding_vectors = []

for item in all_embeddings:

    embedding_vectors.append(

        item["embedding"]

    )

embedding_vectors = np.array(
    embedding_vectors
).astype("float32")

print("\nEmbedding matrix shape:")

print(embedding_vectors.shape)

# =========================
# Create FAISS Index
# =========================

dimension = embedding_vectors.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    embedding_vectors
)

print("\nFAISS index created!")

# =========================
# Save FAISS Index
# =========================

faiss.write_index(

    index,

    "document_index.faiss"

)

print("FAISS index saved!")

# =========================
# Save Metadata
# =========================

with open(
    "metadata.pkl",
    "wb"
) as f:

    pickle.dump(
        all_embeddings,
        f
    )

print("Metadata saved!")

# =========================
# DONE
# =========================

print("\nDONE SUCCESSFULLY!")
