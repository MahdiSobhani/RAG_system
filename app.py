from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import streamlit as st

@st.cache_resource
def create_rag_chain():

    HF_API_KEY = "Your_API_Key"
    embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", huggingfacehub_api_token=HF_API_KEY)

    vectorstore = Chroma(collection_name='monolog' ,persist_directory="monologDB" ,embedding_function=embeddings )                               # load VectorDB

    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": 0.40,"k": 3})  #score_threshold :: only upper 70%           #k :: only top 3 chunck

    GROQ_API_KEY = "Your_API_Key"
    llm = ChatGroq(model="openai/gpt-oss-20b",api_key=GROQ_API_KEY  ,temperature=0 , max_tokens=1000)

    def format_docs(docs):                                           # convert documnet(docs) to text
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template("""
    تو یک دستیار هوشمند هستی که فقط بر اساس Context پاسخ می‌دهد.

    قوانین:

    1. فقط از اطلاعات موجود در Context استفاده کن.
    2. هیچ اطلاعاتی را از دانش قبلی خودت اضافه نکن.
    3. اگر پاسخ سؤال به طور کامل یا کافی در Context وجود ندارد،
    بگو: «اطلاعات کافی در اختیار ندارم.»
    4. پاسخ را با لحن و ادبیات خود متن بده.
    5. لحن و ادبیات متن اصلی را حفظ کن.
    6. اگر پاسخ شامل چند نکته است، آنها را جداگانه بیان کن.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    rag_chain = ({"context": retriever | format_docs, 
                "question": RunnablePassthrough() } | prompt | llm | StrOutputParser())
    
    return rag_chain

rag_chain = create_rag_chain()