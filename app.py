import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import os 

load_dotenv()

st.set_page_config(page_title="Website Agent", page_icon="🤖" )
st.title("Website Agent")

@st.cache_data
def get_pdf_text(url):
    """
    Loads content from a predefined URL and processes it into a string.
    """
    loader = WebBaseLoader(url)
    documents = loader.load()
    text = "\n\n".join([doc.page_content for doc in documents])
    return text


@st.cache_data
def get_text_chunks(text):
    """
    Splits the loaded text into chunks for embedding and retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


@st.cache_resource
def get_vector_store(text_chunks):
    """
    Embeds the text chunks into a vector store for similarity search.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_response(user_query, chat_history, vector_store):
    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation and the document provided:

    Context: {context}
    Chat history: {chat_history}
    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(
    base_url = "https://api.groq.com/openai/v1",
    model_name = "llama-3.3-70b-versatile",
    temperature=1,
    max_tokens=1024
    )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    docs = vector_store.similarity_search(user_query)

    context = "\n".join(doc.page_content for doc in docs)

    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "user_question": user_query,
    })  

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am Website Agent. How can I help you?"),
    ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Type your message here...")

with st.sidebar:
        st.sidebar.header("📝 Enter the URL")
        st.sidebar.write("Provide a URL and ask a question based on the document's content.")
        url = st.sidebar.text_input("Enter the URL to extract context from:", 
    placeholder="e.g., https://en.wikipedia.org/wiki/Harry_Potter")
if url:
    try:
        raw_text = get_pdf_text(url)
    
        text_chunks = get_text_chunks(raw_text)
        
        st.session_state.vector_store = get_vector_store(text_chunks)
        
        st.sidebar.success("Document chargé avec succès !")
    except Exception as e:
        st.sidebar.error(f"Erreur de chargement : {e}")


if user_query is not None and user_query != "": 
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query,unsafe_allow_html=True)

    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            response = (get_response(user_query, st.session_state.chat_history, st.session_state.vector_store))
            st.write(response)
    st.session_state.chat_history.append(AIMessage(content=response))

if url is None :
    st.write("Please enter your URL website before starting the chat.")


        