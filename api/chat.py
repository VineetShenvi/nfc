import os
from flask import Flask, request, jsonify
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from flask_cors import CORS
from prompt import *

load_dotenv()



# Setup Langchain with OpenAI and FAISS
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0.9, max_tokens=2000, model="gpt-3.5-turbo-instruct")

new_vector_store = FAISS.load_local(
    "faiss_index", embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True
)

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=new_vector_store.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
)

app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST', 'GET'])
def chat():
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Perform the query using the RAG model
    result = qa({"query": query})
    return jsonify({"result": result["result"]})


    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
