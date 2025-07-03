from transformers import pipeline
from flask import Flask, request, jsonify
import os

MODEL_DIR = "./serbian-mt5-qa-model"

app = Flask(__name__)

if os.path.exists(MODEL_DIR):
    qa_pipeline = pipeline("text2text-generation",
                           model=MODEL_DIR, tokenizer=MODEL_DIR)
else:
    qa_pipeline = pipeline("text2text-generation",
                           model="google/mt5-base", tokenizer="google/mt5-base")


@app.route("/qa", methods=["POST"])
def answer_question():
    data = request.json
    question = data.get("question")
    context = data.get("context")
    if not question or not context:
        return jsonify({"error": "Missing question or context"}), 400
    input_text = f"question: {question} context: {context}"
    result = qa_pipeline(input_text, max_length=64,
                         clean_up_tokenization_spaces=True)
    answer = result[0]["generated_text"] if result and "generated_text" in result[0] else ""
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
