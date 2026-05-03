import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()


DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/"

def create_vector_db():
    print(f"--- 1. Loading PDFs from {DATA_PATH} ---")
    # Check if directory exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created {DATA_PATH} folder. Please put your PDFs inside and run this again.")
        return

    # Load all PDFs in the data directory
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        print("No PDFs found! Please add PDF files to the 'data' folder.")
        return

    print(f"Loaded {len(documents)} pages.")

    print("--- 2. Splitting Text ---")
    # Split text into chunks of 500 characters with 50 character overlap
    # This helps the bot understand context better
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print("--- 3. Creating Embeddings (Using CPU) ---")
    # using 'all-MiniLM-L6-v2' which is free and runs locally
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )

    print("--- 4. Saving to FAISS Vector Database ---")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    print(f"Success! Database saved to '{DB_FAISS_PATH}'. You can now run app.py.")

if __name__ == "__main__":
    create_vector_db()