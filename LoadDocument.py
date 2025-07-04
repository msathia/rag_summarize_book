from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Update this path to your document directory
DOCS_DIR = "<YOUR DIRECTORY PATH>"

def validate_directory(docs_dir):
    """Validate and fix directory path"""
    if os.path.exists(docs_dir):
        return docs_dir
    
    # Try without trailing space
    docs_dir_trimmed = docs_dir.rstrip()
    if os.path.exists(docs_dir_trimmed):
        return docs_dir_trimmed
    
    print(f"❌ Directory not found: {docs_dir}")
    return None

def load_documents(docs_dir):
    """Load documents from directory"""
    if not validate_directory(docs_dir):
        return []
    
    documents = []
    supported_files = ('.pdf', '.txt', '.docx', '.html', '.md')
    
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.lower().endswith(supported_files):
                file_path = os.path.join(root, file)
                try:
                    loader = UnstructuredFileLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")
    
    return documents

def create_chunks(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    
    return text_splitter.split_documents(documents)

if __name__ == "__main__":
    print("Loading documents from directory...")
    
    # Load and process documents
    documents = load_documents(DOCS_DIR)
    if documents:
        chunks = create_chunks(documents)
        print(f"✅ Loaded {len(documents)} documents, created {len(chunks)} chunks")
    else:
        print("❌ No documents loaded. Check your DOCS_DIR path.")