from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import os

def load_documents_from_directory(docs_dir="<YOUR_DOCUMENTS_DIRECTORY>"):
    """Load and chunk documents from specified directory"""
    if not os.path.exists(docs_dir):
        print(f"❌ Directory {docs_dir} does not exist")
        return []
    
    documents = []
    supported_extensions = ('.pdf', '.txt', '.docx', '.html', '.md')
    
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.lower().endswith(supported_extensions):
                file_path = os.path.join(root, file)
                try:
                    loader = UnstructuredFileLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")
    
    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Loaded {len(documents)} documents, created {len(chunks)} chunks")
    return chunks

def create_embeddings(chunks, persist_directory="chroma_db_binder"):
    """Generate embeddings and store in ChromaDB"""
    if not chunks:
        print("❌ No chunks to process")
        return None
    
    # Initialize embedding model
    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize ChromaDB
    vectorstore = Chroma(
        collection_name="my_binder_collection",
        embedding_function=embeddings_model,
        persist_directory=persist_directory
    )
    
    # Add chunks in batches
    batch_size = 1000
    print(f"Adding {len(chunks)} chunks to ChromaDB...")
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Processing batches"):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
    
    print(f"✅ ChromaDB created at: {persist_directory}")
    return vectorstore

if __name__ == "__main__":
    # Load documents and create embeddings
    chunks = load_documents_from_directory()
    if chunks:
        create_embeddings(chunks)
    else:
        print("No documents to process. Update the DOCS_DIR path.")