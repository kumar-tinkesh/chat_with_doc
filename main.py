import os
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import warnings

warnings.filterwarnings("ignore")

try:
    load_dotenv()
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error during environment setup: {e}")
    st.write("Oops! An error occurred during environment setup.")


def log_event(event):
    print(f"[LOG] {event}")


def get_text_from_pdf(uploaded_file):
    log_event("Reading text from PDF")
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")  # Print error to console
        st.error(f"Oops! An error occurred while reading the file: {e}")  # Show error in Streamlit app
    return text


def get_text_chunks(text):
    log_event("Splitting text into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    log_event("Creating vector store")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    print("embedding is created")


def answer():
    log_event("Initializing question answering model")
    prompt_template = """
    You are an AI assistant that provides helpful answers to user queries.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer: 
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question, text_chunks):
    log_event("Processing user input")
    chat_history_file = "chat_history.txt"
    if not os.path.exists(chat_history_file):
        mode = "w"  # If the file does not exist, open it in write mode
    else:
        mode = "a"  # If the file exists, open it in append mode

    with open(chat_history_file, mode) as f:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        vectordb = FAISS.load_local("faiss_index", embeddings)
        search = vectordb.similarity_search(user_question)

        ans = answer()

        response = ans(
            {"input_documents": search, "question": user_question}
            , return_only_outputs=True)

        if response:
            # Extract text from the response dictionary
            answer_text = response['output_text']
            # Write the chat history (question and answer) to the file
            f.write(f"Question: {user_question}\n")
            f.write(f"Answer: {answer_text}\n\n")
        else:
            answer_text = "Not found in document."
            # Write the chat history (question and answer) to the file
            f.write(f"Question: {user_question}\n")
            f.write(f"Answer: {answer_text}\n\n")

    # Print a log message indicating that the answer has been processed and printed
    log_event("Answer printed")

    print("Answer:", answer_text)

    return clean_text(answer_text), answer_text


def clean_text(output_text):
    # Remove leading and trailing whitespace
    cleaned_text = output_text.strip()

    cleaned_text = '\n'.join(line for line in cleaned_text.splitlines() if line.strip())

    return cleaned_text


def main():
    try:
        log_event("Starting the program")
        # st.title("ISO Document Question Answering")

        default_pdf_path = "docs/ISO+13485-2016.pdf"

        # Check if a file is uploaded, otherwise, use the default PDF file
        uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
        if uploaded_file is None:
            uploaded_file = open(default_pdf_path, 'rb')

        if uploaded_file is not None:
            st.write('ISO File uploaded successfully.')
            text = get_text_from_pdf(uploaded_file)
            chunk = get_text_chunks(text)
            vector = get_vector_store(chunk)
            # st.write('Vector is created.')

            user_question = st.text_input("Ask a Question from document")
            if st.button("Get Answer"):
                answer_text, raw_answer = user_input(user_question, chunk)
                st.write("**Answer:**")
                st.write(answer_text)
                print(answer_text)

        show_chat_history = st.button("Show Chat History")
        if show_chat_history:
            st.subheader("Chat History")
            with open("chat_history.txt", "r") as f:
                chat_history = f.read()
                st.text(chat_history)
    except Exception as e:
        print(f"An error occurred: {e}")
        st.write("Oops! An error occurred.")

if __name__ == "__main__":
    main()
