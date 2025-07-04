from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Configuration - Update this path to your document directory
DOCS_DIR = "<YOUR_GOOGLE_DRIVE_PATH>"
GOOGLE_DRIVE_FOLDER_ID = "YOUR_FOLDER_ID_HERE"  # Replace with actual folder ID

def load_google_docs():
    """Load Google Docs from Drive"""
    try:
        from langchain_google_community import GoogleDriveLoader
        
        google_loader = GoogleDriveLoader(
            folder_id=GOOGLE_DRIVE_FOLDER_ID,
            recursive=True,
            file_types=["document", "sheet"],
        )
        
        google_docs = google_loader.load()
        print(f"✅ Loaded {len(google_docs)} Google Docs")
        return google_docs
        
    except ImportError:
        print("❌ langchain-google-community not installed")
        print("Install with: pip install langchain-google-community[drive]")
        return []
    except Exception as e:
        print(f"❌ Google Drive error: {e}")
        return []

def load_local_files(docs_dir):
    """Load local files from directory"""
    if not os.path.exists(docs_dir):
        print(f"❌ Directory not found: {docs_dir}")
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
    
    print(f"✅ Loaded {len(documents)} local files")
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
    print("Loading documents from local directory and Google Drive...")
    
    # Load from both sources
    documents = []
    documents.extend(load_google_docs())
    documents.extend(load_local_files(DOCS_DIR))
    
    if documents:
        chunks = create_chunks(documents)
        google_count = len([d for d in documents if 'google' in d.metadata.get('source', '').lower()])
        local_count = len(documents) - google_count
        
        print(f"\n✅ Summary:")
        print(f"   • Google Docs: {google_count}")
        print(f"   • Local files: {local_count}")
        print(f"   • Total chunks: {len(chunks)}")
    else:
        print("❌ No documents loaded") 