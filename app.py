from flask import Flask, request, jsonify
from retriever import retrieve_context
from transformers import pipeline



app = Flask(__name__)

agent_pipeline = pipeline("question-answering", model="deepset/xlm-roberta-large-squad2")



@app.route("/qa", methods=["POST"])
def answer_question():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing question"}), 400

    # Retrieve context from vector DB
    context = retrieve_context(question)
    print (f"Retrieved context: {context}")
    if not context:
        return jsonify({"error": "No relevant context found"}), 404
    #Prepare context for input
    context_text = """"""
    if context["type"] == "qa":
       context_text = f"""{context['answer']}"""

    if context["type"] == "product":
        context_text = f"""
        Naziv proizvoda: {context['name']}
        Opis proizvoda: {context['description']}
        SKU: {context['sku']}
        URL: www.shoppy.rs/{context['url_key']}/
        """


    # Generate output tokens
    result = agent_pipeline({
        "question": question,
        "context": context_text
    })

    # Decode to string

    print (f"Model receives: {context_text} and outputs: {result}")
    answer = result["answer"]

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
