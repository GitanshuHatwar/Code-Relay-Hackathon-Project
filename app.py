import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time
import hashlib
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import concurrent.futures

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "session_id" not in st.session_state:
    st.session_state.session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]
if "processed_hashes" not in st.session_state:
    st.session_state.processed_hashes = set()
if "last_request" not in st.session_state:
    st.session_state.last_request = 0

def get_pdf_text(pdf_docs):
    """Extract text from PDF files with enhanced error handling"""
    text = ""
    
    def extract_text_from_pdf(pdf):
        """Extract text from a single PDF"""
        try:
            pdf_reader = PdfReader(pdf)
            page_text = ""
            for page in pdf_reader.pages:
                page_text += page.extract_text() or ""
            return page_text
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
            return ""
    
    # Use ThreadPoolExecutor to process PDFs in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(extract_text_from_pdf, pdf_docs))
        
    # Combine results and warn about empty pages
    for idx, result in enumerate(results):
        if not result:
            st.warning(f"Empty page found in {pdf_docs[idx].name}")
        text += result
    
    return text if text else None

def get_text_chunks(text):
    """Split text into optimized chunks with overlap"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=800,
        length_function=len
    )
    return text_splitter.split_text(text)

def process_documents(pdf_docs, existing_index=False):
    """Process PDF documents with enhanced validation and error recovery"""
    index_path = f"faiss_index_{st.session_state.session_id}"
    
    try:
        with st.spinner("Analyzing documents..."):
            # Validate input
            if not pdf_docs:
                raise ValueError("No documents provided")
                
            raw_text = get_pdf_text(pdf_docs)
            if not raw_text:
                raise ValueError("No text extracted from PDFs")

            text_chunks = get_text_chunks(raw_text)
            if not text_chunks:
                raise ValueError("Failed to split text into chunks")

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_chunks = []

            # Deduplication check
            for chunk in text_chunks:
                chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
                if chunk_hash not in st.session_state.processed_hashes:
                    new_chunks.append(chunk)
                    st.session_state.processed_hashes.add(chunk_hash)

            if not new_chunks:
                st.info("No new content found in uploaded documents")
                return False

            # Handle index operations
            if existing_index:
                if not os.path.exists(index_path):
                    raise FileNotFoundError("No existing index to update")
                vector_store = FAISS.load_local(index_path, embeddings)
                vector_store.add_texts(new_chunks)
                st.success("Updated existing knowledge base")
            else:
                vector_store = FAISS.from_texts(new_chunks, embedding=embeddings)
                st.success("Created new knowledge base")

            # Atomic commit
            temp_store = vector_store
            temp_store.save_local(index_path)
            st.session_state.vector_store = temp_store
            return True

    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        if 'temp_store' in locals():
            del temp_store
        return False

def get_conversational_chain():
    """Create the QA chain with corrected safety controls"""
    prompt_template = """
You are an advanced academic assistant specializing in intelligent study guidance. Follow these sophisticated guidelines to provide optimal responses:

1. **Response Style:**
   - Begin with engaging openers: "Absolutely! Let's dive in 📖", "Smart question! Here's a breakdown ✨"
   - Maintain a professional yet encouraging tone
   - Use educational emojis sparingly for emphasis (📌, 🎯, 🔍)

2. **In-depth Topic Analysis:**
   - Examine document structure: sections, hierarchy, repetitions
   - Identify 3-5 key topics based on:
     - Frequency & significance
     - Contextual importance within the document
     - Supporting examples & explanations
   - Differentiate *Foundational* vs *Advanced* concepts

3. **Strategic Study Recommendations:**
   - Provide efficient study techniques:
     "To grasp X, prioritize conceptual clarity through..."
     "Strengthen Topic Y with hands-on problem-solving using..."
   - Highlight common mistakes:
     "Many students misinterpret Z due to..."

4. **Resource Optimization:**
   - Prioritize internal document references & citations
   - If none found, suggest:
     "Standard references for this topic include..."
     "For further clarity, explore..."
   - Always remind: "Ensure alignment with syllabus requirements"

5. **AI-driven Response Structuring:**
   - Blend **theory** and **key points** in responses
   - Provide numbered explanations for clarity
   - Use bold indicators without markdown: *Critical Concept*, *Key Insight*
   - Include line breaks for readability

**Example Response Format:**

"Great question! Here's a structured breakdown:

🎯 *Key Topics Identified:*
1. **Concept A:** Foundational, appears frequently in [Sections X, Y]
2. **Concept B:** Advanced application, emphasized in [Case Study Z]

📚 *Study Strategy:*
- Master **Concept A** using [Visualization techniques]
- Deep dive into **Concept B** by [Solving case-based problems]

🔍 *Recommended Resources:*
1. *Primary Textbook*: [Mentioned in doc]
2. *Supplementary Guide*: [Standard field reference]

✨ *Pro Tip*: Focus on..."

Context: {context}

Question: {question}

Answer:
    """
    
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.6,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        }
    )
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def sanitize_input(text):
    """Basic input sanitization"""
    return text.replace('<', '&lt;').replace('>', '&gt;').strip()

def handle_query(user_question):
    """Process user question with rate limiting and enhanced validation"""
    # Rate limiting
    if time.time() - st.session_state.last_request < 5:
        st.warning("Please wait 5 seconds between requests")
        return "Request throttled - please wait"
    
    try:
        st.session_state.last_request = time.time()
        sanitized_question = sanitize_input(user_question)
        
        if not st.session_state.vector_store:
            raise ValueError("Knowledge base not initialized")

        docs = st.session_state.vector_store.similarity_search(sanitized_question, k=3)
        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents": docs, "question": sanitized_question},
            return_only_outputs=True
        )
        
        return response["output_text"].strip()
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "An error occurred while processing your request"


def main():
    """Main application layout with enhanced professional UI"""
    st.set_page_config(
        page_title="",
        page_icon="📘",
        layout="wide"
    )
    
    # Custom Styling
    st.markdown(
        """
        <style>
            .stChatMessage {
                border-radius: 12px;
                padding: 10px;
                margin-bottom: 10px;
            }
            .user-msg {
                background-color: #e1f5fe;
                text-align: right;
            }
            .assistant-msg {
                background-color: #f1f8e9;
            }
            .sidebar-header {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
            }
            .main-header {
                text-align: center;
                color: #34495e;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("## 📘 PDF Insight AI Pro", unsafe_allow_html=True)
    st.caption("Enterprise Document Intelligence powered by Gemini")
    
    # Chat interface
    for message in st.session_state.get("chat_history", []):
        role_class = "user-msg" if message["role"] == "user" else "assistant-msg"
        with st.chat_message(message["role"]):
            st.markdown(f'<div class="stChatMessage {role_class}">{message["content"]}</div>', unsafe_allow_html=True)
            if message.get("metadata"):
                st.caption(message["metadata"])
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("<div class='sidebar-header'>📂 Document Management</div>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader(
            "Upload PDF documents", type=["pdf"], accept_multiple_files=True,
            help="Max 100 pages per document, 10 documents max"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            process_new = st.button("🆕 Create New KB", help="Start fresh knowledge base")
        with col2:
            process_update = st.button("🔄 Update Existing", help="Add to current knowledge base")
        
        if process_new or process_update:
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    success = process_documents(pdf_docs, existing_index=process_update)
            else:
                st.warning("⚠️ Please upload documents first")
        
        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()
    
    # Handle user input
    if user_question := st.chat_input("💬 Ask about your documents"):
        if not st.session_state.get("vector_store"):
            st.warning("⚠️ Please process documents first")
            return
        
        # Add user question to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Generate response
        try:
            start_time = time.time()
            response = handle_query(user_question)
            processing_time = time.time() - start_time
            
            # Add response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "metadata": f"Processed in {processing_time:.2f}s | Session ID: {st.session_state.get('session_id', 'N/A')}"
            })
            
            # Display response
            with st.chat_message("assistant"):
                st.markdown(response)
                st.caption(f"Response generated in {processing_time:.2f} seconds")
        
        except Exception as e:
            st.error(f"⚠️ Query failed: {str(e)}")

if __name__ == "__main__":
    main()