# 📚 RAG Playground - Document Summarization System

A powerful **Retrieval-Augmented Generation (RAG)** system for document summarization and question-answering using ChromaDB, Ollama, and LangChain.

## 🎯 Features

- **📄 Multi-format Document Support**: PDF, DOCX, TXT, HTML, Markdown
- **🔍 Intelligent Document Chunking**: Smart text splitting for optimal retrieval
- **🧠 Local LLM Integration**: Powered by Ollama for privacy and speed
- **💾 Persistent Vector Storage**: ChromaDB for efficient similarity search
- **🌐 Google Drive Integration**: Load documents directly from Google Drive
- **🎨 Interactive Chat Interface**: Natural language queries with source citations

## 🏗️ Architecture

```
Documents → Chunking → Embeddings → ChromaDB → RAG Chain → LLM → Answers
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Git**
- **Homebrew** (macOS) or equivalent package manager

### 1. Clone the Repository

```bash
git clone https://github.com/msathia/rag_summarize_book.git
cd rag_summarize_book
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv my_rag_env

# Activate virtual environment
source my_rag_env/bin/activate  # macOS/Linux
# or
my_rag_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Install System Dependencies

```bash
# Install Ollama (Local LLM)
brew install ollama

# Install Tesseract (OCR for PDFs)
brew install tesseract

# Install Poppler (PDF processing)
brew install poppler

# Install ChromaDB system dependencies (if needed)
# ChromaDB is installed via pip, but may need system libraries
brew install sqlite3
```

### 4. Install Python Dependencies

```bash
# Install all required Python packages
pip install -r requirements.txt

# This includes:
# - chromadb==1.0.13 (Vector database)
# - langchain packages (RAG framework)
# - sentence-transformers (Embeddings)
# - unstructured (Document processing)
# - and more...
```

### 5. Set Up Ollama

```bash
# Start Ollama service
ollama serve

# In another terminal, pull a model
ollama pull llama3.2
# or
ollama pull mistral
```

### 6. Load Your Documents

```bash
# Load documents from local directory
python LoadDocument.py

# Or load from Google Drive (requires setup)
python LoadDocumentWithGoogleDrive.py
```

### 7. Generate Embeddings

```bash
# Create embeddings and store in ChromaDB
python GenerateEmbeddings.py
```

### 8. Start Chat Interface

```bash
# Interactive chat with your documents
python ChatInterface.py
```

## 📁 Project Structure

```
rag_summarize_book/
├── 📄 LoadDocument.py              # Local document loader
├── ☁️ LoadDocumentWithGoogleDrive.py  # Google Drive integration
├── 🧠 GenerateEmbeddings.py        # Embedding generation
├── 💬 ChatInterface.py             # Interactive chat interface
├── 🔍 RAGQueryLogic.py             # Query processing logic
├── 📋 SetupDevEnv.txt              # Development setup guide
├── 📖 README.md                    # This file
├── 🚫 .gitignore                   # Git ignore rules
├── 📚 chroma_db_binder/            # Vector database (auto-created)
└── 🐍 my_rag_env/                  # Python virtual environment
```

## 🔧 Configuration

### ChromaDB Setup

ChromaDB is automatically installed via `pip install -r requirements.txt`. The database is created locally in the `chroma_db_binder/` directory.

```python
# In your code, ChromaDB is initialized like this:
persist_directory = "chroma_db_binder"
vectorstore = Chroma(
    collection_name="my_binder_collection",
    embedding_function=embeddings_model,
    persist_directory=persist_directory
)
```

**Database Location**: `./chroma_db_binder/` (auto-created)

### Document Loading

Edit `LoadDocument.py` to specify your document directory:

```python
DOCS_DIR = "/path/to/your/documents"
```

### Google Drive Setup

1. Follow instructions in `GOOGLE_DRIVE_SETUP.md`
2. Update `GOOGLE_DRIVE_FOLDER_ID` in `LoadDocumentWithGoogleDrive.py`

### Model Configuration

The system automatically detects available Ollama models:
- `llama3.2`
- `llama3.2:1b`
- `mistral`
- `codellama`
- `phi3`

## 💡 Usage Examples

### Basic Document Query

```bash
python ChatInterface.py
# Choose option 1 for chat interface
# Ask: "What are the main topics covered in this document?"
```

### Test Retrieval System

```bash
python ChatInterface.py
# Choose option 2 to test retrieval
```

### Load Specific Document Types

```python
# In LoadDocument.py, modify the file filter:
if file.lower().endswith(('.pdf', '.txt', '.docx', '.html', '.md')):
    # Process document
```

## 🛠️ Troubleshooting

### Common Issues

**❌ "Ollama not available"**
```bash
# Make sure Ollama is running
ollama serve
# In another terminal
ollama run llama3.2
```

**❌ "Tesseract not installed"**
```bash
brew install tesseract
```

**❌ "Poppler not installed"**
```bash
brew install poppler
```

**❌ "ChromaDB connection error"**
```bash
# Remove and recreate database
rm -rf chroma_db_binder/
python GenerateEmbeddings.py
```

### Performance Tips

- **Use smaller models** (`llama3.2:1b`) for faster responses
- **Adjust chunk size** in `LoadDocument.py` for your use case
- **Batch processing** for large document collections
- **Monitor memory usage** during embedding generation

## 🔒 Privacy & Security

- **Local Processing**: All document processing happens on your machine
- **No Cloud Dependencies**: Ollama runs locally, no data sent to external services
- **Secure Storage**: ChromaDB stores vectors locally
- **API Key Protection**: `.gitignore` prevents accidental key commits

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the RAG framework
- **ChromaDB** for vector storage
- **Ollama** for local LLM hosting
- **Sentence Transformers** for embeddings

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/msathia/rag_summarize_book/issues)
- **Discussions**: [GitHub Discussions](https://github.com/msathia/rag_summarize_book/discussions)

---

**Made with ❤️ for document intelligence**

