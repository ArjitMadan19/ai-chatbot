from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load all txt files from docs folder
loader = DirectoryLoader(
    "docs",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={
        "encoding": "utf-8",
        "autodetect_encoding": True
    }
)

documents = loader.load()

print(f"Loaded {len(documents)} documents")


# 2. Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

print(f"Created {len(docs)} chunks")


# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Store embeddings in FAISS
db = FAISS.from_documents(docs, embeddings)


# 5. Save vector database locally
db.save_local("vectorstore")

print("Vector database saved successfully!")