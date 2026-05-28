# =========================================================
# AI HYBRID SEARCH + RAG SYSTEM
# Flask + FAISS + BM25 + Groq + Typo Correction
# =========================================================

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)

import os
import re
import pickle
import traceback
import numpy as np
import faiss

from groq import Groq

from sentence_transformers import SentenceTransformer

from rapidfuzz import process, fuzz

from sklearn.preprocessing import normalize

from rank_bm25 import BM25Okapi


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

# =========================================================
# DOCUMENTS FOLDER
# =========================================================

DOCUMENTS_FOLDER = "sample"

# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading embedding model...\n")

model = SentenceTransformer(

    "all-MiniLM-L6-v2",

    device="cpu"

)

print("Embedding model loaded!")

# =========================================================
# GROQ CLIENT
# =========================================================

GROQ_API_KEY = os.getenv("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")
client = Groq( api_key=GROQ_API_KEY )

# =========================================================
# LOAD FAISS
# =========================================================

print("\nLoading FAISS index...\n")

index = faiss.read_index(
    "document_index.faiss"
)

faiss.omp_set_num_threads(4)

print("FAISS loaded!")

# =========================================================
# LOAD METADATA
# =========================================================

print("\nLoading metadata...\n")

with open("metadata.pkl", "rb") as f:

    metadata = pickle.load(f)

print("Metadata loaded!")

# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# =========================================================
# CREATE VOCABULARY
# =========================================================

print("\nCreating vocabulary...\n")

vocab = set()

for item in metadata:

    text = clean_text(
        item["text"]
    )

    words = text.split()

    for word in words:

        if len(word) > 2:

            vocab.add(word)

vocab = list(vocab)

print(f"Vocabulary size: {len(vocab)}")

# =========================================================
# TYPO CORRECTION
# =========================================================

def correct_query(query):

    corrected = []

    for word in query.lower().split():

        if len(word) <= 2:

            corrected.append(word)

            continue

        match = process.extractOne(

            word,

            vocab,

            scorer=fuzz.ratio

        )

        if match:

            candidate = match[0]

            score = match[1]

            if score >= 80:

                corrected.append(candidate)

            else:

                corrected.append(word)

        else:

            corrected.append(word)

    return " ".join(corrected)

# =========================================================
# SUGGESTIONS
# =========================================================

print("\nCreating suggestions...\n")

suggestions = []

for item in metadata:

    text = clean_text(
        item["text"]
    )

    words = text.split()

    limit = min(
        len(words) - 3,
        12
    )

    for i in range(limit):

        phrase = " ".join(
            words[i:i+4]
        )

        if len(phrase) > 8:

            suggestions.append({

                "text": phrase,

                "file": item["file"]

            })

# =========================================================
# REMOVE DUPLICATES
# =========================================================

unique = {}

for item in suggestions:

    text = item["text"]

    if text not in unique:

        unique[text] = item

suggestions = list(
    unique.values()
)

suggestions = suggestions[:30000]

print(f"Suggestions: {len(suggestions)}")

# =========================================================
# SUGGESTION EMBEDDINGS
# =========================================================

print("\nCreating suggestion embeddings...\n")

suggestion_texts = [

    item["text"]

    for item in suggestions
]

suggestion_embeddings = model.encode(                                                          # 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

    suggestion_texts,

    batch_size=256,

    convert_to_numpy=True,

    show_progress_bar=True

).astype("float32")

suggestion_embeddings = normalize(
    suggestion_embeddings
)

print("Suggestion embeddings ready!")

# =========================================================
# QUERY EXPANSION
# =========================================================

def expand_query(query):

    query_embedding = model.encode(

        query,

        convert_to_numpy=True

    ).astype("float32")

    query_embedding = normalize(
        [query_embedding]
    )[0]

    similarities = np.dot(

        suggestion_embeddings,

        query_embedding

    )

    top_indices = np.argsort(
        similarities
    )[-5:][::-1]

    expanded_parts = [query]

    used_words = set(
        query.lower().split()
    )

    for idx in top_indices:

        phrase = suggestion_texts[idx]

        phrase_words = set(
            phrase.split()
        )

        overlap = len(
            phrase_words & used_words
        )

        if overlap >= len(phrase_words) * 0.7:

            continue

        expanded_parts.append(
            phrase
        )

        used_words.update(
            phrase_words
        )

        if len(expanded_parts) >= 4:

            break

    expanded_parts = list(
        dict.fromkeys(expanded_parts)
    )

    return " | ".join(expanded_parts)

# =========================================================
# BM25 CORPUS
# =========================================================

print("\nCreating BM25 corpus...\n")

bm25_corpus = []

for item in metadata:

    text = clean_text(
        item["text"]
    )

    bm25_corpus.append(
        text.split()
    )

bm25 = BM25Okapi(
    bm25_corpus
)

print("BM25 ready!")

# =========================================================
# RAG GENERATION
# =========================================================

def generate_rag_answer(

    query,

    retrieved_chunks

):

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are a helpful AI document assistant.

Answer ONLY using the provided context.

If answer is unavailable,
say:
"I could not find relevant information."

================ CONTEXT ================

{context}

=========================================

QUESTION:
{query}

ANSWER:
"""

    try:

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[

                {
                    "role": "system",
                    "content":
                    "You answer using only retrieved documents."
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ],

            temperature=0.2,

            max_tokens=500

        )

        answer = response.choices[0].message.content

        if not answer:

            return "No answer generated."

        return answer

    except Exception:

        traceback.print_exc()

        return "Error generating answer."

# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    results = []

    rag_answer = ""

    corrected_query = ""
    expanded_query = ""
    original_query = ""

    if request.method == "POST":

        query = request.form["query"]

        original_query = query

        # =================================================
        # TYPO CORRECTION
        # =================================================

        corrected_query = correct_query(
            query
        )

        # =================================================
        # QUERY EXPANSION
        # =================================================

        expanded_query = expand_query(
            corrected_query
        )

        final_query = expanded_query

        print("\nOriginal Query:")
        print(original_query)

        print("\nCorrected Query:")
        print(corrected_query)

        print("\nExpanded Query:")
        print(expanded_query)

        # =================================================
        # QUERY EMBEDDING
        # =================================================

        query_embedding = model.encode(

            final_query,

            convert_to_numpy=True

        ).astype("float32")

        query_embedding = np.array(
            [query_embedding]
        )

        faiss.normalize_L2(
            query_embedding
        )

        # =================================================
        # FAISS SEARCH
        # =================================================

        top_k = 50

        similarities, indices = index.search(

            query_embedding,

            top_k

        )

        similarities = similarities[0]

        indices = indices[0]

        # =================================================
        # BM25 SCORES
        # =================================================

        tokenized_query = clean_text(
            final_query
        ).split()

        bm25_scores = bm25.get_scores(
            tokenized_query
        )

        # =================================================
        # HYBRID RESULTS
        # =================================================

        hybrid_results = []

        for i, idx in enumerate(indices):

            if idx >= len(metadata):

                continue

            semantic_score = similarities[i]

            keyword_score = bm25_scores[idx]

            hybrid_score = (

                semantic_score * 0.7 +

                keyword_score * 0.3

            )

            hybrid_results.append({

                "idx": idx,

                "score": hybrid_score

            })

        # =================================================
        # SORT RESULTS
        # =================================================

        hybrid_results = sorted(

            hybrid_results,

            key=lambda x: x["score"],

            reverse=True

        )

        # =================================================
        # REMOVE DUPLICATES
        # =================================================

        shown_files = set()

        retrieved_chunks = []

        rank = 1

        for item in hybrid_results:

            idx = item["idx"]

            result = metadata[idx]

            file_name = result["file"]

            if file_name in shown_files:

                continue

            shown_files.add(file_name)

            retrieved_chunks.append(
                result["text"]
            )

            results.append({

                "rank": rank,

                "file": file_name,

                "text": result["text"][:600],

                "score": round(
                    item["score"],
                    4
                )

            })

            rank += 1

            if rank > 20:

                break

        # =================================================
        # RAG ANSWER
        # =================================================

        rag_answer = generate_rag_answer(

            original_query,

            retrieved_chunks[:5]

        )

    return render_template(

        "index1.html",

        results=results,

        rag_answer=rag_answer,

        corrected_query=corrected_query,

        expanded_query=expanded_query,

        original_query=original_query
    )

# =========================================================
# AUTOCOMPLETE
# =========================================================

@app.route("/autocomplete")
def autocomplete():

    query = request.args.get(
        "q",
        ""
    ).lower()

    if len(query) < 2:

        return jsonify({

            "suggestions": []

        })

    matches = process.extract(

        query,

        suggestion_texts,

        limit=10
    )

    results = []

    for match in matches:

        matched_text = match[0]

        for item in suggestions:

            if item["text"] == matched_text:

                results.append({

                    "text": item["text"],

                    "file": item["file"]

                })

                break

    return jsonify({

        "suggestions": results

    })

# =========================================================
# OPEN DOCUMENT
# =========================================================

@app.route("/open/<path:filename>")
def open_document(filename):

    return send_from_directory(

        os.path.abspath(
            DOCUMENTS_FOLDER
        ),

        filename,

        as_attachment=False
    )

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )


