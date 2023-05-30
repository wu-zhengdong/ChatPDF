import os
import tempfile
import pandas as pd
import streamlit as st
from datetime import datetime
from openai import OpenAIError
from pdfminer.high_level import extract_text

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain


def read_pdf(file):
    text = extract_text(file)
    return text

def get_response(docs, query, model_name='gpt-4', chain_type="stuff"):

    llm = ChatOpenAI(temperature=0.1, model_name=model_name, max_tokens=500)
    chain = load_qa_chain(llm, chain_type=chain_type)
    response = chain.run(input_documents=docs, question=query)
    
    return response

# Start the Streamlit application
st.title('ChatPDF')

# API Key Input
api_key = st.text_input("Please enter your OpenAI API key", type='password')

# If no API Key is provided
if not api_key:
    st.warning("Please provide an OpenAI API key to use this application.")
    st.stop()

# Set the API Key as an environment variable
os.environ["OPENAI_API_KEY"] = api_key

# Default parameters
k = 6
model_name = "gpt-4"
chunk_size = 1000
chunk_overlap = 200

# Default queries
default_queries = [
    "What was the research gap?",
    "What were the objectives of the study?",
    "What was the study design?",
    "How and when the data were collected?",
    "What sampling methods were used to select the study participants?",
    "What do you think of the appropriateness of the sampling methods?",
    "What were the four key outcome measures (dependent variables) of the study?",
    "How were they measured? What were the scales of measurement?",
    "What independent/confounding variables were collected to investigate their relationships with the key outcome measures?",
    "What statistical tests were used to investigate the relationships?",
    "What was the originally planned/required sample size of this study and how was it determined?"
]

# 1. Upload PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# If a PDF has been uploaded
if uploaded_file is not None:
    # read PDF and convert to raw text
    # Save uploaded file to a temporary file, then read it
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    raw_text = read_pdf(tfile.name)
    tfile.close()

    # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    
    # Download embeddings from OpenAI
    try:
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
    except OpenAIError:
        st.warning("The provided OpenAI API key is not valid. Please check and provide a valid key.")
        st.stop()

    # Create a filename for the chat history
    pdf_name = uploaded_file.name.split('.')[0]  # Remove the '.pdf' extension
    current_date = datetime.now().strftime('%Y%m%d')  # Get the current date in 'yyyymmdd' format
    chat_history_filename = f"{pdf_name}_{current_date}.xlsx"

    # Try to read the chat history from the Excel file
    try:
        chat_history = pd.read_excel(chat_history_filename).values.tolist()
    except FileNotFoundError:
        chat_history = []  # If the file does not exist, start with an empty chat history

    # 2. Chat interface
    st.subheader('Chatbot')
    user_input = st.selectbox("Select a default question or type your own:", [""] + default_queries, key='user_input')
    user_input = st.text_input("Or type your own question:", value=user_input, key='custom_input')

    if st.button("Send"):
        if user_input.strip() != "":
            with st.spinner('Generating response...(Please wait 10 to 20 seconds)'):
                docs = docsearch.similarity_search(user_input, k=k)
                response = get_response(docs, query=user_input, model_name=model_name)
            # Save the query and response
            chat_history.insert(0, [user_input, response])
            pd.DataFrame(chat_history, columns=['Query', 'Response']).to_excel(chat_history_filename, index=False)
            # Display the chat history
            for i, (q, r) in enumerate(chat_history):
                st.write(f"You: {q}")
                st.write(f"ChatPDF: {r}")
        else:
            st.warning("Please input some question.")
    else:
        # Display previous chat history even if no new input
        for i, (q, r) in enumerate(chat_history):
            st.write(f"You: {q}")
            st.write(f"ChatPDF: {r}")

    # Export Button
    if st.button("Export"):
        with open(chat_history_filename, "rb") as file:
            btn = st.download_button(
                label="Download Chat History",
                data=file,
                file_name=chat_history_filename,
                mime='application/vnd.ms-excel',
            )