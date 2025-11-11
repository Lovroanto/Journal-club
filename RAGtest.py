from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Create a test file
os.makedirs("test_paper", exist_ok=True)
with open("test_paper/test.txt", "w") as f:
    f.write("Synaptic plasticity is the ability of synapses to strengthen or weaken over time.")

# Load and embed
loader = TextLoader("test_paper/test.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(chunks, embeddings)

# Create LLM
llm = OllamaLLM(model="llama3.1")

retriever = db.as_retriever()
query = "What is synaptic plasticity?"
docs = retriever.invoke(query)
context = "\n".join([d.page_content for d in docs])

print("AI:", llm.invoke(f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"))