# RAG_system
A Persian Retrieval-Augmented Generation (RAG) application that answers questions based only on the content of a PDF document.

# 📚 Persian RAG Assistant

A Persian **Retrieval-Augmented Generation (RAG)** application that answers questions based only on the content of a PDF document.

The project combines **Hugging Face Embeddings**, **Chroma Vector Database**, **LangChain**, **Groq**, and **Streamlit** to build a simple end-to-end RAG system.

## ✨ Features

* Persian question answering over PDF content
* Semantic document retrieval using vector embeddings
* Persistent Chroma vector database
* Similarity score threshold for filtering irrelevant chunks
* Top-k document retrieval
* Context-aware LLM generation
* Prevents the LLM from using external knowledge outside the retrieved context
* Persian RTL Streamlit interface
* Chat-style conversation UI
* Cached RAG chain for better Streamlit performance

## 🏗️ Architecture

```text
                PDF Document
                     │
                     ▼
              Document Loading
                     │
                     ▼
              Text Chunking
                     │
                     ▼
              Hugging Face
                Embeddings
                     │
                     ▼
             Chroma Vector DB
                     │
                     │
User Question ───────┘
      │
      ▼
   Embedding
      │
      ▼
   Retriever
      │
      ├── similarity_score_threshold
      ├── score_threshold = 0.40
      └── k = 3
      │
      ▼
 Retrieved Chunks
      │
      ▼
    Context
      │
      ▼
   Prompt Template
      │
      ▼
  GPT-OSS-20B
      │
      ▼
    Answer
      │
      ▼
   Streamlit UI
```

## 🧠 RAG Pipeline

### 1. Embeddings

The project uses:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

through Hugging Face's embedding interface.

This model is suitable for multilingual semantic representation and is used to convert both documents and user queries into vectors.

```python
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    huggingfacehub_api_token=HF_API_KEY
)
```

### 2. Vector Database

The project uses **Chroma** as the vector database.

The vector database is persisted locally:

```text
monologDB/
```

and loaded when the application starts.

```python
vectorstore = Chroma(
    collection_name="monolog",
    persist_directory="monologDB",
    embedding_function=embeddings
)
```

### 3. Retrieval

Instead of always returning the top `k` chunks, the project uses a similarity threshold:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.40,
        "k": 3
    }
)
```

This means:

* `k = 3` → at most 3 relevant chunks are retrieved.
* `score_threshold = 0.40` → chunks below the relevance threshold are filtered out.

This helps prevent unrelated documents from being passed to the LLM.

### 4. Context Formatting

Retrieved documents are converted into a single context string:

```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```

### 5. Prompt Engineering

The LLM is explicitly instructed to answer only from the retrieved context.

Important rules include:

```text
- Only use information available in the Context.
- Do not add information from previous knowledge.
- If the answer is not sufficiently available in the Context,
  respond with: "اطلاعات کافی در اختیار ندارم."
- Preserve the tone and writing style of the original text.
```

This creates a controlled RAG pipeline and reduces unsupported answers.

### 6. LLM

The project uses:

```text
openai/gpt-oss-20b
```

through Groq.

Configuration:

```python
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=1000
)
```

`temperature=0` is used to make the generated responses more deterministic.

### 7. RAG Chain

The complete chain is built with LangChain:

```python
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

The flow is:

```text
Question
   ↓
Retriever
   ↓
Relevant Chunks
   ↓
Context Formatting
   ↓
Prompt
   ↓
LLM
   ↓
String Output
```

## 🖥️ Streamlit Interface

The project includes a simple Persian chat interface built with Streamlit.

Features:

* RTL Persian layout
* Chat history
* User/assistant message separation
* Loading indicator
* Text input through `st.chat_input`
* Conversation state through `st.session_state`

The UI is implemented in `stream.py`.

## 📁 Project Structure

```text
RAG-Project/
│
├── app.py
├── stream.py
├── RAG.ipynb
├── monologDB/
└── README.md
```

### `RAG.ipynb`

Used for the RAG development and vector database preparation.

### `app.py`

Contains:

* Embedding model
* Chroma vector database
* Retriever
* Prompt
* LLM
* Complete RAG chain

The RAG chain is cached with Streamlit's `@st.cache_resource`.

### `stream.py`

Contains the Streamlit user interface and invokes the RAG chain for each user question.

### `monologDB/`

Persistent Chroma vector database containing the document embeddings.

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install dependencies:

```bash
pip install langchain
pip install langchain-chroma
pip install langchain-huggingface
pip install langchain-groq
pip install streamlit
```

## 🔑 Environment Variables

API keys should be stored as environment variables rather than directly inside Python source code.

For example:

```text
HF_API_KEY=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
```

Then load them in Python:

```python
import os

HF_API_KEY = os.getenv("HF_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

> **Security:** Never commit API keys to GitHub. If a real API key has already been committed to a repository, revoke/rotate it and replace it with an environment variable.

## ▶️ Run the Application

Run the Streamlit application:

```bash
streamlit run stream.py
```

Then open the local Streamlit URL shown in the terminal.

## 💬 Example

User:

```text
این متن درباره چه موضوعی صحبت می‌کند؟
```

The system:

```text
Question
   ↓
Semantic Search
   ↓
Relevant Chunks
   ↓
Context
   ↓
LLM
```

If the required information exists in the document, the assistant generates an answer based on the retrieved context.

If sufficient information cannot be found:

```text
اطلاعات کافی در اختیار ندارم.
```

## 🛠️ Technologies

| Technology            | Purpose                      |
| --------------------- | ---------------------------- |
| Python                | Programming language         |
| LangChain             | RAG pipeline orchestration   |
| Chroma                | Vector database              |
| Hugging Face          | Text embeddings              |
| Sentence Transformers | Multilingual embedding model |
| Groq                  | LLM inference                |
| GPT-OSS-20B           | Language model               |
| Streamlit             | Web UI                       |

## 🎯 Main Concepts Demonstrated

This project demonstrates an end-to-end RAG workflow including:

* Document processing
* Text chunking
* Embeddings
* Vector databases
* Semantic search
* Similarity score thresholding
* Top-k retrieval
* Prompt engineering
* Context injection
* LLM generation
* Hallucination control through context restriction
* LangChain Runnable pipelines
* Streamlit integration

## 🚀 Future Improvements

Possible extensions:

* [ ] Add source citations to answers
* [ ] Display retrieved chunks and similarity scores
* [ ] Improve chunking strategy
* [ ] Evaluate different embedding models
* [ ] Add metadata filtering
* [ ] Add conversation-aware retrieval
* [ ] Add retrieval evaluation metrics
* [ ] Add document upload directly from Streamlit
* [ ] Support multiple documents
* [ ] Add hybrid search
* [ ] Add reranking
* [ ] Add automated RAG evaluation

## 📌 Project Goal

The goal of this project is to demonstrate a practical **Persian RAG system** that connects semantic retrieval with an LLM while restricting the model to information retrieved from the source document.

The project focuses on understanding the complete RAG pipeline rather than simply calling an LLM:

```text
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Database
    ↓
Retriever
    ↓
Context
    ↓
Prompt
    ↓
LLM
    ↓
Answer
```
