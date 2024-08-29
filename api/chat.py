import os
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify
import langchain_openai 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from flask_cors import CORS
import openai
from prompt import *

load_dotenv()



# Setup Langchain with OpenAI and FAISS
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

llm = langchain_openai.OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0.9, max_tokens=2000, model="gpt-3.5-turbo-instruct")
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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

temp_folder = tempfile.mkdtemp()

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

@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    print("hi")
    try:
        audio_file = request.files['audio']
        if audio_file and audio_file.filename.endswith(('.mp3', '.wav', '.flac', '.mp4')):
            audio_path = os.path.join(temp_folder, audio_file.filename)
            audio_file.save(audio_path)
            transcript = client.audio.translations.create(
                model="whisper-1",
                file=Path(audio_path),
                response_format="text"
            )
            return jsonify({"transcript": transcript})

        else:
            return jsonify({"error": "Invalid audio file format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
