from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from google import genai
import os
import time

app = Flask(__name__)

# ---- Step 1: Configure Gemini client (used for BOTH embeddings and generation) ----
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.6-flash"

CONFIDENCE_THRESHOLD = 0.6
TOP_K = 3

# ---- Step 2: Load and clean data ----
print("Loading dataset...")
data = pd.read_excel("ParentGuidence.xlsx")
data = data.dropna(subset=["Questions", "Answers"])
data = data[
    (data["Questions"].str.strip() != "") &
    (data["Answers"].str.strip() != "")
]
questions = data["Questions"].tolist()
answers = data["Answers"].tolist()


def embed_text(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text
            )
            return np.array(result.embeddings[0].values)
        except Exception as e:
            print(f"Embedding attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---- Step 3: Precompute embeddings ----
CACHE_FILE = "question_embeddings.npy"

if os.path.exists(CACHE_FILE):
    print("Loading cached embeddings from disk (no API calls needed)...")
    question_embeddings = np.load(CACHE_FILE)
    print(f"Loaded {len(question_embeddings)} cached embeddings.")
else:
    print("No cache found. Encoding questions via Gemini API...")
    question_embeddings = []
    for i, q in enumerate(questions):
        emb = embed_text(q)
        question_embeddings.append(emb)
        time.sleep(1.5)
        print(f"Embedded {i+1}/{len(questions)}")
    question_embeddings = np.array(question_embeddings)
    np.save(CACHE_FILE, question_embeddings)
    print(f"Saved embeddings to {CACHE_FILE} for future runs.")

print("Chatbot is ready! (lightweight mode - no local ML model loaded)")


@app.route("/chat", methods=["POST"])
def chat():
    user_question = request.json.get("question", "")
    if user_question.strip() == "":
        return jsonify({"answer": "Please ask a valid question."})


    user_embedding = embed_text(user_question)

    scores = np.array([
        cosine_similarity(user_embedding, q_emb) for q_emb in question_embeddings
    ])

    top_indices = scores.argsort()[::-1][:TOP_K]
    top_scores = scores[top_indices]

    if top_scores[0] >= CONFIDENCE_THRESHOLD:
        context_blocks = []
        for idx in top_indices:
            context_blocks.append(f"Q: {questions[idx]}\nA: {answers[idx]}")
        context_text = "\n\n".join(context_blocks)
    else:
        context_text = "(no relevant parenting information found for this message)"

    prompt = f"""You are a helpful, friendly parenting assistant.

First, check the type of message:
- If the user's message is just a greeting, introduction, or casual small talk (like "hi",
  "hello", "thanks", "my name is X") — reply in ONE short, warm sentence (max 15 words).
  If they mentioned their name, greet them by name (e.g. "Hi Ahmad! How can I help you
  with parenting today?").
- If the user is asking a real parenting question or wants advice/suggestions about their
  child, use the context below to answer in 2-3 clear, easy-to-understand sentences —
  not just one line, and not a long essay. Briefly explain the "why," not just the "what,"
  so it feels genuinely helpful. Write it as one smooth, coherent paragraph, not stitched
  fragments. If the context doesn't fully answer the question, say so honestly instead of
  making something up.
- If it's a real parenting question but the context has no relevant information, politely
  say you don't have information on that specific topic, in one short sentence.

Context:
{context_text}

User message: {user_question}

Answer:"""

    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )
        generated_answer = response.text
    except Exception as e:
        print(f"Generation error: {e}")
        generated_answer = "I'm having trouble generating a response right now — please try again in a moment."

    return jsonify({
        "answer": generated_answer,
        "matched_score": float(top_scores[0])
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )