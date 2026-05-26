from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------
# 1. Load all txt files
# -----------------------------
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


# -----------------------------
# 2. Add custom metadata
# -----------------------------
def get_doc_type(file_path):
    path = Path(file_path)

    if "contracts" in path.parts:
        return "contract"
    elif "research_papers" in path.parts:
        return "research_paper"
    elif "notes" in path.parts:
        return "notes"
    else:
        return "unknown"


def clean_title(file_path):
    filename = Path(file_path).stem
    return filename.replace("_", " ").replace("-", " ").title()


for doc in documents:
    source_path = doc.metadata.get("source", "")

    doc.metadata["doc_type"] = get_doc_type(source_path)
    doc.metadata["title"] = clean_title(source_path)
    doc.metadata["file_name"] = Path(source_path).name
    doc.metadata["folder"] = Path(source_path).parent.name

print("\nDocument metadata preview:")
for doc in documents:
    print(doc.metadata)


# -----------------------------
# 3. Split documents into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

print(f"\nCreated {len(docs)} chunks")


# -----------------------------
# 4. Add chunk-level metadata
# -----------------------------
for i, doc in enumerate(docs):
    doc.metadata["chunk_id"] = i


# -----------------------------
# 5. Create embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# 6. Store in FAISS
# -----------------------------
db = FAISS.from_documents(docs, embeddings)


# -----------------------------
# 7. Save vector database
# -----------------------------
db.save_local("vectorstore")

print("Vector database saved successfully!")