import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.memory import ConversationalBufferMemory
from langchain.chains import ConversationalRetrievalChain

from InstructorEmbedding import INSTRUCTOR


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    model = INSTRUCTOR('hkunlp/instructor-xl')
    embeddings = model.encode(text_chunks)


    embeddings = model.encode(text_chunks)

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    memory = ConversationalBufferMemory(memory_key='chat-history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(

    )


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    st.header("Chat with Multiple PDFs :books:")
    st.text_input("Ask questions about the PDFs")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click process", type="pdf", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):

                # Step 1: Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # Step 2: Get PDF text chunks
                text_chunks = get_text_chunks(raw_text)
               # st.write(text_chunks)

                # Step 3: Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create Conversation Chain
                conversation = get_conversation_chain(vectorstore)
                


if __name__ == '__main__':
    main()