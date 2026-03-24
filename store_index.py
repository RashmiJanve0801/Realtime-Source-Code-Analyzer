from src.helper import repo_ingestion, load_repo, text_splitter, load_embedding
from langchain_community.vectorstores import Chroma
import os

from dotenv import load_dotenv
load_dotenv()


os.environ["GOOGLE_API_KEY"] =  os.environ.get("GOOGLE_API_KEY")

documents = load_repo("repo/")
text_chunks = text_splitter(documents)
embeddings = load_embedding()

# Storing vector in chromadb
vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory='./db')

