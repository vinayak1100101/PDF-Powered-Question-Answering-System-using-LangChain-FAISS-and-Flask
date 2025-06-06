from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from flask import *

load_dotenv()
app=Flask(__name__)
# Set your OpenAI API key here or export OPENAI_API_KEY env var instead
OPENAI_API_KEY =  os.getenv('OPENAI_API_KEY')
# Path to FAISS index folder
FAISS_INDEX_PATH = "sample_faiss"

# 1. Loading PDF document
loader = PyPDFLoader("example.pdf")
documents = loader.load()

# 2. Spliting into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 3. Prepare embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Load or create vectorstore
if os.path.exists(FAISS_INDEX_PATH):
    print("Loading vectorstore from disk...")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("Creating vectorstore and saving to disk...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)

# 5. Create retriever and QA chain
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY),
    retriever=retriever,
)

# 6. Query example
# query = "can you give brief description about what pygame will do"
# response = qa.run(query)
# print("Answer:", response)




@app.route('/')
def home():
    return render_template('input.html')

@app.route('/input', methods=['POST'])
def input_route():
    input_text = request.form['input_text']
    #input query
    query = input_text
    response = qa.run(query)
    output_response = response

    return redirect(url_for('output', response=output_response))

@app.route('/output')
def output():
    response = request.args.get('response', '')
    return render_template('output.html', output_response=response)


if __name__=='__main__':
    app.run(debug=True)
