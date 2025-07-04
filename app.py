from flask import Flask, request, jsonify
import os
from retriever import retrieve_context
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import tensorflow as tf

MODEL_DIR = "./serbian-mt5-qa-model"

app = Flask(__name__)

if os.path.exists(MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
else:
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    model = TFAutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")


@app.route("/qa", methods=["POST"])
def answer_question():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing question"}), 400

    # Retrieve context from vector DB
    context = retrieve_context(question)
    input_text = f"question: {question} context: {context if context else ''}"

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="tf", truncation=True, padding=True)

    # Generate output tokens
    outputs = model.generate(**inputs, max_new_tokens=64)

    # Decode to string
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    print("Server is running on http://0.0.0.0:5000")
    print("Use Ctrl+C to stop the server")
