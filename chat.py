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

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_text_from_pdf(pdf_file):
    text = ""
    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
    except FileNotFoundError:
        st.error(f"File '{pdf_file}' not found.")
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def answer():
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


def user_input(user_question):
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
            
    return clean_text(answer_text), answer_text


def clean_text(output_text):
    # Remove leading and trailing whitespace
    cleaned_text = output_text.strip()
    
    cleaned_text = '\n'.join(line for line in cleaned_text.splitlines() if line.strip())

    return cleaned_text


def main():
    print("Loading PDF file...")
    st.title("ISO Document Question Answering")
    
    file = "docs/ISO+13485-2016.pdf"
    print(f"Loaded file: {file}")
    
    text = get_text_from_pdf(file)
    print("PDF file processed. extracting text...")
    
    chunk = get_text_chunks(text)
    print("Text extracted. Splitting text into chunks...")
    
    vector = get_vector_store(chunk)
    print("Text chunks processed. vector store created...")
    
    st.write('vector is created')

    user_question = st.text_input("Ask a Question from the ISO document")
    if st.button("Get Answer"):
        print("User submitted question...")
        answer_text, raw_answer = user_input(user_question)
        print("Answer obtained.")
        st.write("**Answer:**")
        st.write(answer_text)
        
        print("Answer   :", raw_answer)
        
    if st.button("Display Chat History"):
        display_chat_history()


def display_chat_history():
    chat_history_file = "chat_history.txt"
    if os.path.exists(chat_history_file):
        with open(chat_history_file, "r") as f:
            st.write("Chat History:")
            st.code(f.read())
    else:
        st.write("Chat history not found.")


if __name__ == "__main__":
    main()