from transformers import pipeline
from flask import Flask, request, jsonify
import os
from retriever import retrieve_context

MODEL_DIR = "./serbian-mt5-qa-model"

app = Flask(__name__)

if os.path.exists(MODEL_DIR):
    qa_pipeline = pipeline("text2text-generation",
                           model=MODEL_DIR, tokenizer=MODEL_DIR, framework="tf")
else:
    qa_pipeline = pipeline("text2text-generation",
                           model="google/mt5-small", tokenizer="google/mt5-small", framework="tf")


@app.route("/qa", methods=["POST"])
def answer_question():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing question"}), 400
    
    # Step 1: Try to answer as FAQ (no context)
    input_text = f"question: {question} context: "
    result = qa_pipeline(input_text, max_length=64, clean_up_tokenization_spaces=True)
    answer = result[0]["generated_text"] if result and "generated_text" in result[0] else ""

    # Step 2: If the answer is empty or generic, try with retrieved context
    if not answer or answer.strip().lower() in ["", "i don't know", "unknown", "n/a", "not found", "no answer"]:
        context = retrieve_context(question)
        input_text = f"question: {question} context: {context if context else ''}"
        result = qa_pipeline(input_text, max_length=64, clean_up_tokenization_spaces=True)
        answer = result[0]["generated_text"] if result and "generated_text" in result[0] else ""
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    print("Server is running on http://0.0.0.0:5000")
    print("Use Ctrl+C to stop the server")
