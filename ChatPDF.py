'''
With database
'''
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

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Function to update the Google sheet
def update_google_sheet(database_name_email, query, response):
    # Get the current date and time
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")

    # Try to open the spreadsheet
    try:
        sheet = client.open(database_name_email)
    except gspread.SpreadsheetNotFound:
        st.warning(f"No existing Google Sheet found with the name '{database_name_email}'.")
        st.stop()

    # Select the first worksheet in the spreadsheet
    worksheet = sheet.get_worksheet(0)
    # Then append the row to the selected worksheet
    worksheet.append_row([query, response, current_time])

def read_pdf(file):
    text = extract_text(file)
    return text

def get_response(docs, query, model_name, chain_type="stuff"):

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

# Connect to Google Sheets
scope = ['https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive"]

credentials = ServiceAccountCredentials.from_json_keyfile_name("chatpdf-388607-557f6bf3d25a.json", scope)
client = gspread.authorize(credentials)

st.subheader('1. Please provide the Gmail address and spreadsheet name.')
# Get the email and spreadsheet name from the user
user_email = st.text_input("Please enter your Gmail address", value="")

# Validate the inputs
if not user_email.endswith("@gmail.com"):
    st.warning("Please enter a valid Gmail address.")
    # st.stop()

database_name = st.text_input("Please enter your spreadsheet name", value="")

# Combine the spreadsheet name and email to create a unique name
database_name_email = "{}_{}".format(database_name, user_email.split("@")[0])

create_new_sheet = st.checkbox("Create a new Google Sheet?", value=True)

# If user chose to create a new sheet
if create_new_sheet:
    if st.button("Create and Send Google Sheet"):
        try:
            # Create a new sheet and share to user's Google excel
            sheet = client.create(database_name_email)
            sheet.share(user_email, perm_type='user', role='writer')
            
            # Select the first worksheet in the spreadsheet
            worksheet = sheet.get_worksheet(0)

            # Check if the headers are already there, if not, create them
            headers = worksheet.row_values(1)
            if not headers:
                worksheet.append_row(["query", "response", "time"])

            st.warning(f"{database_name_email} sheet created and linked successfully!")
        except Exception as e:
            st.error(f"Could not create a new Google Sheet: {str(e)}")
            st.stop()
else:
    if st.button("Please press this button to verify the existence of the Google Sheet."):
        try:
            sheet = client.open(database_name_email)
            st.warning(f"Your Google Drive hosts the {database_name_email} sheet. Connection successful!")
        except gspread.SpreadsheetNotFound:
            st.warning(f"No existing Google Sheet found with the name '{database_name_email}'.")
            st.stop()

st.subheader('2. Please choose the parameters, model, and upload the PDF file.')
# Default parameters
chunk_size = 1000
chunk_overlap = 200
k = st.selectbox("Choose a larger number to expand your search and retrieve more information.", options=[4, 6, 8, 12, 16], index=1)
model_name = st.selectbox("Choose a model", options=["gpt-4", "gpt-3.5-turbo"], index=0)

# Default queries
default_queries = [
    "What was the research gap?",
    "What were the objectives of the study?",
    "What was the study design?",
    "How and when the data were collected?",
    "What is the time span of this dataset?",
    # "What sampling methods were used to select the study participants?",
    # "What do you think of the appropriateness of the sampling methods?",
    "What were the outcome measures (dependent variables) of the study?",
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
    st.subheader('3. Please select a question or type your own question.')
    user_input = st.selectbox("Select a default question:", [""] + default_queries, key='user_input')
    user_input = st.text_input("Or type your own question:", value=user_input, key='custom_input')

    if st.button("Send"):
        if user_input.strip() != "":
            with st.spinner('Generating response...(Please wait around 10 to 30 seconds)'):
                docs = docsearch.similarity_search(user_input, k=k)
                response = get_response(docs, query=user_input, model_name=model_name)
                update_google_sheet(database_name_email, user_input, response)
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
    st.write('**You can also click ***EXPORT*** to save the chat history to your local PC.**')
    if st.button("EXPORT"):
        with open(chat_history_filename, "rb") as file:
            btn = st.download_button(
                label="Download chat history",
                data=file,
                file_name=chat_history_filename,
                mime='application/vnd.ms-excel',
            )