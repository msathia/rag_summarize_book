import os
import re
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

def setup_rag_system():
    """Set up the RAG system by loading the existing ChromaDB"""
    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        collection_name="my_binder_collection",
        embedding_function=embeddings_model,
        persist_directory="chroma_db_binder"
    )
    
    print(f"✅ Loaded vector database with {vectorstore._collection.count()} chunks")
    return vectorstore

def setup_llm():
    """Set up the language model"""
    ollama_models = ["llama3.2", "llama3.2:1b", "llama3", "mistral", "codellama", "llama2", "phi3"]
    
    for model_name in ollama_models:
        try:
            llm = Ollama(model=model_name, base_url="http://localhost:11434")
            llm.invoke("Hello")  # Test connection
            print(f"✅ Using Ollama model: {model_name}")
            return llm
        except Exception:
            continue
    
    print("❌ No Ollama models available. Start Ollama and pull a model.")
    return None

def detect_question_type(question):
    """Smart routing: Detect if question is document-related or generic"""
    
    # Document-related keywords
    doc_keywords = [
        'document', 'file', 'paper', 'report', 'contract', 'agreement', 'policy',
        'according to', 'what does', 'mentioned', 'stated', 'written', 'contains',
        'section', 'chapter', 'page', 'appendix', 'clause', 'article',
        'compliance', 'requirements', 'specifications', 'guidelines', 'procedures'
    ]
    
    # Generic question patterns
    generic_patterns = [
        r'^(what|how|why|when|where|who)\s+(is|are|do|does|can|should|would|will)',
        r'^(explain|define|describe|tell me about)',
        r'^(what\'s|how\'s|when\'s)',
        r'(weather|time|date|cook|recipe|sports|news|celebrity|movie|music)',
        r'^(calculate|solve|compute)',
        r'(general|common|typical|usual|normal|standard)\s+(practice|approach|method)'
    ]
    
    question_lower = question.lower()
    
    # Check for generic patterns first (more specific)
    for pattern in generic_patterns:
        if re.search(pattern, question_lower):
            return "generic"
    
    # Check for document keywords
    if any(keyword in question_lower for keyword in doc_keywords):
        return "document"
    
    # If question mentions "my", "our", "this", "these" - likely document-related
    if re.search(r'\b(my|our|this|these|the)\s+\w+', question_lower):
        return "document"
    
    # Default to document mode for ambiguous cases
    return "document"

def create_document_chain(llm, vectorstore):
    """Create a RAG chain optimized for document questions"""
    prompt_template = """You are a helpful assistant specialized in analyzing documents.

Use the following pieces of context from the documents to answer the question at the end.

INSTRUCTIONS:
1. If the context contains relevant information, provide a detailed answer citing specific documents.
2. If the context lacks relevant information, say: "I don't have specific information about this in the available documents."
3. Always cite source document(s) by filename when using specific information.
4. Focus on information directly from the documents.

Context:
{context}

Question: {question}

Answer based on documents:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=False
    )

def create_general_chain(llm):
    """Create a general Q&A chain for non-document questions"""
    prompt_template = """You are a helpful AI assistant. Answer the following question using your general knowledge.

Be helpful, accurate, and concise. If you're not sure about something, say so.

Question: {question}

Answer:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    
    return LLMChain(llm=llm, prompt=prompt, verbose=False)

def create_hybrid_chain(llm, vectorstore):
    """Create a hybrid chain that can handle both document and general questions"""
    prompt_template = """You are a helpful AI assistant that can answer questions using both document context and general knowledge.

Use the following pieces of context from documents if relevant to the question. If the context is not relevant or the question is general knowledge, answer using your general knowledge.

INSTRUCTIONS:
1. If the context is relevant to the question, use it and cite sources.
2. If the context is not relevant or the question is about general topics (like cooking, weather, sports, etc.), provide a helpful general answer.
3. Be transparent about whether you're using document information or general knowledge.

Context from documents:
{context}

Question: {question}

Helpful Answer:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=False
    )

def chat_with_mode_selection():
    """Chat interface with explicit mode selection"""
    print("🎯 Advanced Document Q&A System")
    print("\nChoose your interaction mode:")
    print("1. 📄 Document Mode - Focus on your document collection")
    print("2. 🌐 General Mode - Ask any questions freely") 
    print("3. 🔄 Auto Mode - Smart routing based on question type")
    print("4. 🔀 Hybrid Mode - Combined document + general knowledge")
    
    mode_choice = input("\nSelect mode (1-4): ").strip()
    
    # Set up components
    vectorstore = setup_rag_system()
    llm = setup_llm()
    
    if llm is None:
        return
    
    # Create appropriate chain based on mode
    if mode_choice == "1":
        chain = create_document_chain(llm, vectorstore)
        mode_name = "📄 Document Mode"
        print(f"\n✅ {mode_name} activated - Focusing on your documents")
    elif mode_choice == "2":
        chain = create_general_chain(llm)
        mode_name = "🌐 General Mode"
        print(f"\n✅ {mode_name} activated - Free Q&A mode")
    elif mode_choice == "3":
        # Auto mode - we'll handle routing in the loop
        doc_chain = create_document_chain(llm, vectorstore)
        general_chain = create_general_chain(llm)
        mode_name = "🔄 Auto Mode"
        print(f"\n✅ {mode_name} activated - Smart question routing")
    elif mode_choice == "4":
        chain = create_hybrid_chain(llm, vectorstore)
        mode_name = "🔀 Hybrid Mode"
        print(f"\n✅ {mode_name} activated - Document + general knowledge")
    else:
        print("Invalid choice. Using Auto Mode...")
        doc_chain = create_document_chain(llm, vectorstore)
        general_chain = create_general_chain(llm)
        mode_name = "🔄 Auto Mode"
        mode_choice = "3"
    
    print("Type 'quit', 'exit', or 'bye' to end.\n")
    
    while True:
        user_question = input("🤔 You: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'bye', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_question:
            continue
        
        try:
            # Handle Auto Mode with smart routing
            if mode_choice == "3":
                question_type = detect_question_type(user_question)
                print(f"🤖 Detected: {question_type} question")
                
                if question_type == "document":
                    result = doc_chain.invoke({"query": user_question})
                    print(f"\n📄 Document Assistant: {result['result']}")
                    
                    if result.get('source_documents'):
                        sources = [os.path.basename(doc.metadata.get('source', 'Unknown')) 
                                  for doc in result['source_documents'][:3]]
                        print(f"📚 Sources: {', '.join(sources)}")
                else:
                    result = general_chain.invoke({"question": user_question})
                    print(f"\n🌐 General Assistant: {result['text']}")
                    print("📚 Source: General knowledge")
            
            # Handle General Mode
            elif mode_choice == "2":
                result = chain.invoke({"question": user_question})
                print(f"\n🌐 Assistant: {result['text']}")
                print("📚 Source: General knowledge")
            
            # Handle Document, Hybrid Modes
            else:
                result = chain.invoke({"query": user_question})
                print(f"\n🤖 Assistant: {result['result']}")
                
                if result.get('source_documents'):
                    sources = [os.path.basename(doc.metadata.get('source', 'Unknown')) 
                              for doc in result['source_documents'][:3]]
                    print(f"📚 Sources: {', '.join(sources)}")
                else:
                    print("📚 Source: General knowledge")
            
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}\n")

def simple_chat():
    """Simple chat interface (backward compatibility)"""
    print("🎯 Document Q&A Chat Interface")
    print("Ask questions about your document collection!")
    print("Type 'quit', 'exit', or 'bye' to end.\n")
    
    vectorstore = setup_rag_system()
    llm = setup_llm()
    
    if llm is None:
        return
    
    qa_chain = create_hybrid_chain(llm, vectorstore)
    
    while True:
        user_question = input("🤔 You: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'bye', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_question:
            continue
        
        try:
            result = qa_chain.invoke({"query": user_question})
            print(f"\n🤖 Assistant: {result['result']}")
            
            if result.get('source_documents'):
                sources = [os.path.basename(doc.metadata.get('source', 'Unknown')) 
                          for doc in result['source_documents'][:3]]
                print(f"📚 Sources: {', '.join(sources)}")
            else:
                print("📚 Source: General knowledge")
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}\n")

def test_retrieval():
    """Test the retrieval system"""
    vectorstore = setup_rag_system()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    test_questions = ["document content", "data processing", "vendor security", "compliance requirements"]
    
    for question in test_questions:
        docs = retriever.get_relevant_documents(question)
        preview = docs[0].page_content[:100].replace('\n', ' ') if docs else "No content"
        print(f"'{question}': {len(docs)} chunks found - {preview}...")

def test_smart_routing():
    """Test the smart routing functionality"""
    print("🧪 Testing Smart Question Routing\n")
    
    test_cases = [
        ("What does my contract say about payment terms?", "document"),
        ("How do I cook pasta?", "generic"),
        ("What's the weather like today?", "generic"),
        ("According to the policy document, what are the requirements?", "document"),
        ("Explain machine learning", "generic"),
        ("What are the compliance requirements mentioned?", "document"),
        ("How to solve this math problem?", "generic"),
        ("What sections does this document contain?", "document")
    ]
    
    for question, expected in test_cases:
        detected = detect_question_type(question)
        status = "✅" if detected == expected else "❌"
        print(f"{status} '{question[:50]}...' -> {detected} (expected: {expected})")

if __name__ == "__main__":
    print("🎯 Advanced Document Q&A RAG System\n")
    print("1. Advanced Chat (Mode Selection)")
    print("2. Simple Chat (Hybrid Mode)")
    print("3. Test Retrieval System")
    print("4. Test Smart Routing")
    
    choice = input("\nChoose (1-4): ").strip()
    
    if choice == "1":
        chat_with_mode_selection()
    elif choice == "2":
        simple_chat()
    elif choice == "3":
        test_retrieval()
    elif choice == "4":
        test_smart_routing()
    else:
        print("Invalid choice. Starting advanced chat...")
        chat_with_mode_selection() 