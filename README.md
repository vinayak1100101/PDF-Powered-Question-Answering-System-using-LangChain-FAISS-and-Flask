﻿PDF-Powered Q&A System with LangChain and Flask
This project is a web-based, question-answering application that allows you to chat with your PDF documents. It uses a Retrieval-Augmented Generation (RAG) pipeline built with LangChain to understand a PDF's content and answer your questions about it. The application is served through a simple Flask web interface.

Features ✨
Interactive QA Web App: A user-friendly web interface built with Flask to ask questions and receive answers.

PDF Document Processing: Ingests and processes any PDF document using PyPDFLoader.

Advanced Text Chunking: Employs RecursiveCharacterTextSplitter to intelligently split the document into manageable chunks for analysis.

Efficient Text Embeddings: Utilizes HuggingFace's all-MiniLM-L6-v2 model to create vector embeddings from the text chunks.

Vector-Based Retrieval: Leverages a FAISS (Facebook AI Similarity Search) vector store to quickly find and retrieve the most relevant document chunks for a given query.

Powerful Language Generation: Integrates OpenAI's GPT-4o-mini model to generate human-like, context-aware answers.

Persistent Vector Store: Saves the created FAISS index to disk, avoiding the need for re-computation on subsequent runs and improving startup time.

How It Works ⚙️
The application follows a Retrieval-Augmented Generation (RAG) architecture:

Load & Split: The example.pdf document is loaded and split into smaller, overlapping text chunks.

Embed & Store: Each chunk is converted into a numerical vector (embedding) using a HuggingFace model. These embeddings are stored in a FAISS vector store, which acts as a searchable knowledge base. This index is saved locally in the sample_faiss folder to speed up future launches.

Retrieve: When you ask a question, your query is also converted into an embedding. The application then searches the FAISS index to find the text chunks with embeddings most similar to your query's embedding.

Generate: The retrieved chunks (the relevant context) and your original question are passed to the GPT-4o-mini model. The model then generates a coherent answer based on the provided information.

Serve: The entire process is wrapped in a Flask application, which provides a simple web UI for you to enter your question and view the generated answer.

🛠️ Tech Stack
Backend Framework: Flask

Orchestration Framework: LangChain

LLM: OpenAI GPT-4o-mini

Embedding Model: HuggingFace all-MiniLM-L6-v2

Vector Store: FAISS (from langchain-community)

PDF Loading: PyPDFLoader (from langchain-community)

Environment Variables: python-dotenv

🚀 Getting Started
Follow these instructions to set up and run the project locally.

Prerequisites
Python 3.8 or higher

An OpenAI API Key

1. Clone the Repository
git clone <your-repository-url>
cd <repository-directory>

2. Create a Virtual Environment
It's recommended to use a virtual environment to manage project dependencies.


# For Windows
python -m venv venv
.\venv\Scripts\activate

3. Install Dependencies
You will need to create a requirements.txt file. Add the following contents to it:

flask
langchain
langchain-community
langchain-openai
langchain-huggingface
faiss-cpu
pypdf
python-dotenv

Then, install the packages:

pip install -r requirements.txt

4. Set Up Environment Variables
Create a file named .env in the root directory of the project and add your OpenAI API key:

OPENAI_API_KEY="your-openai-api-key-here"

5. Add Your PDF
Place the PDF file you want to query in the root directory and name it example.pdf. If you wish to use a different PDF, make sure to update the filename in main.py:

# in main.py
loader = PyPDFLoader("your-file-name.pdf")

🏃‍♀️ Usage
Run the Flask Application:
Execute the main.py script from your terminal.

python main.py

The first time you run it, it will create and save the FAISS vector store. Subsequent runs will be faster as it will load the index from the disk.

Access the Web App:
Open your web browser and navigate to:

http://127.0.0.1:5000

Ask a Question:
You will see an input form. Type your question about the PDF document and submit it. The application will process your query and display the answer on the next page.

📂 Project Structure
A typical structure for this project would be:

.
├── templates/
│   ├── input.html      # HTML for the input form
│   └── output.html     # HTML to display the response
├── sample_faiss/       # Directory where the FAISS index is stored
├── .env                # Stores environment variables (e.g., API keys)
├── main.py             # Main Flask application and RAG pipeline logic
├── example.pdf         # The PDF document to be queried
└── requirements.txt    # Project dependencies
