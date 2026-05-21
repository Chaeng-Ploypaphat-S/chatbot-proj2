from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pdfplumber
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

st.header("Here is my 2nd chat bot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload your regulatory documents here (pdf only):", type="pdf")

# Input box always visible
user_question = st.text_input("Type your question here")

@st.cache_resource
def process_file(file_bytes):
    import io
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", "", "."],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=st.secrets["OPENAI_API_KEY"]
    )

    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

if file is not None:
    vector_store = process_file(file.read())  # cached after first run

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=1000,
        openai_api_key=st.secrets["OPENAI_API_KEY"]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a PDF document.\n\n"
         "Guidelines:\n"
         "1. Provide complete, well-explained answers using the context below.\n"
         "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
         "3. If the context mentions related information, include it to give fuller picture.\n"
         "4. Only use information from the provided context - do not use outside knowledge.\n"
         "5. Summarize long information, ideally in bullets where needed\n"
         "6. If the information is not in the context, say so politely.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if user_question:
        response = chain.invoke(user_question)
        st.write(response)