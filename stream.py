from app import rag_chain ,st

st.set_page_config(page_title="RAG Assistant",page_icon="📚")

st.markdown("""
<style>
.block-container{max-width:900px;padding-top:5rem}
.stMarkdown{direction:rtl;text-align:right}
</style>
""",unsafe_allow_html=True)

st.title("📚 RAG Assistant")
st.caption("پرسش خود را درباره محتوای PDF وارد کنید.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query=st.chat_input("پیام خود را بنویسید...")

if query:
    st.session_state.messages.append({"role":"user","content":query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("در حال فکر کردن..."):
            answer=rag_chain.invoke(query)
        st.markdown(answer)
    st.session_state.messages.append({"role":"assistant","content":answer})

# streamlit run app.py