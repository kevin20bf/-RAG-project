import streamlit as st
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

prompt_template = """Answer the following question based only on the provided context:
<context>
    {context}
</context>
<question>
    {input}
</question>
"""

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def main():
    st.set_page_config(page_title="RAG", layout="wide")
    st.subheader("Retrieval Augmented generation", divider="blue")

    with st.sidebar:
        st.sidebar.title("Data loader")

        pdf_docs = st.file_uploader(label="Load your pdfs", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Loading"):
                content = ""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        content += page.extract_text()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=512, chunk_overlap=16
                )

                chunks = splitter.split_text(content)
                st.write(chunks)

                embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                vector_store = Chroma.from_texts(
                    chunks,
                    embedding_model,
                    collection_name="data_collection",
                )
                retriever = vector_store.as_retriever(
                    search_kwargs={"k": 5},
                )

                st.session_state.retriever = retriever

    st.subheader("Chatbot")
    user_question = st.text_input("Ask Your Question")
    if user_question:
        context_docs = st.session_state.retriever.invoke(user_question)
        context_list = [d.page_content for d in context_docs]
        context_text = ". ".join(context_list)
        prompt = prompt_template.format(context=context_text, input=user_question)

        resp = llm.invoke(prompt)

        st.write(resp.content)


if __name__ == "__main__":
    main()
