# Placeholder for product/database retriever
# Replace with actual retrieval logic (DB, search, etc.)

# Example product database (replace with real DB or search)
PRODUCTS = {
    "product x": "Product X costs 5000 dinars and is available in red and blue.",
    "product y": "Product Y is out of stock.",
}

def retrieve_context(question):
    q = question.lower()
    # Simple heuristic: if question mentions a known product, return its info
    for product, info in PRODUCTS.items():
        if product in q:
            return info
    # If not a product question, return None (FAQ or general question)
    return None
