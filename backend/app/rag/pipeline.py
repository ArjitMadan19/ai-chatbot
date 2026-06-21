from pathlib import Path
from threading import Lock

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import CrossEncoder
from transformers import pipeline

from backend.app.rag.memory import select_conversation_history_for_query
from backend.app.services.config import settings


FALLBACK_ANSWER = "I do not know based on the provided documents."
SELF_CORRECTION_MARKERS = [
    "\nWait,",
    "\nWait.",
    "\nI need to correct",
    "\nCorrection:",
    "\nActually,",
    "\nLet me correct",
    "Wait,",
    "Wait.",
    "I need to correct",
    "Correction:",
    "Actually,",
    "Let me correct",
]

ROUTE_EXAMPLES = {
    "memory_transform": [
        "Rewrite the previous answer.",
        "Summarize the previous answer.",
        "Make the previous answer shorter.",
        "Explain the previous answer in more detail.",
        "Give one key point from the previous answer.",
        "Convert the previous answer into bullet points."
    ],
    "followup_retrieve": [
        "Ask for more factual details about the same document topic.",
        "Ask who is responsible for something mentioned earlier.",
        "Ask where the previous topic is mentioned in the document.",
        "Ask about exceptions, obligations, dates, or parties related to the previous answer."
    ],
    "new_retrieve": [
        "Ask a new question about a document.",
        "Ask what a contract says about a topic.",
        "Ask what a research paper explains.",
        "Ask a standalone question that needs document search."
    ],
    "chat": [
        "Say thanks.",
        "Acknowledge the answer.",
        "End the conversation.",
        "Casual conversation."
    ]
}

# -----------------------------
# 1. Load embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model_name
)


# -----------------------------
# 2. Load FAISS vector database
# -----------------------------
db = FAISS.load_local(
    settings.vectorstore_dir,
    embeddings,
    allow_dangerous_deserialization=settings.allow_dangerous_deserialization
)
vectorstore_write_lock = Lock()

reranker = CrossEncoder(settings.reranker_model_name)
# -----------------------------
# 3. Retriever
# -----------------------------
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": settings.retrieval_k,
        "fetch_k": settings.retrieval_fetch_k
    }
)


# -----------------------------
# 4. Load local Hugging Face LLM
# -----------------------------

pipe = pipeline(
    "text-generation",
    model=settings.llm_model_name,
    max_new_tokens=settings.llm_max_new_tokens,
    do_sample=False,
    return_full_text=False
)


# -----------------------------
# 7. Conversation memory
# -----------------------------
chat_history = []
MAX_MEMORY_TURNS = 3


class DocumentLoadError(Exception):
    pass


class LLMGenerationError(Exception):
    pass


class VectorStoreError(Exception):
    pass


def clean_title(file_path):
    filename = Path(file_path).stem
    return filename.replace("_", " ").replace("-", " ").title()


def load_document(file_path):
    path = Path(file_path)

    if path.suffix.lower() == ".txt":
        loader = TextLoader(
            str(path),
            encoding="utf-8",
            autodetect_encoding=True
        )
    elif path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        raise ValueError("Only .txt and .pdf uploads are supported.")

    try:
        return loader.load()
    except Exception as error:
        raise DocumentLoadError(f"Could not read document: {path.name}") from error


def add_uploaded_document(file_path, doc_type):
    """
    Loads one uploaded document, adds metadata, indexes it in FAISS, and persists the vectorstore.
    """

    documents = load_document(file_path)
    path = Path(file_path)

    for doc in documents:
        doc.metadata["doc_type"] = doc_type
        doc.metadata["title"] = clean_title(path)
        doc.metadata["file_name"] = path.name
        doc.metadata["folder"] = path.parent.name
        doc.metadata["source"] = str(path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)

    next_chunk_id = db.index.ntotal
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = next_chunk_id + index

    if chunks:
        try:
            with vectorstore_write_lock:
                db.add_documents(chunks)
                db.save_local(settings.vectorstore_dir)
        except Exception as error:
            raise VectorStoreError("Could not add document to the vector database.") from error

    return {
        "file_name": path.name,
        "doc_type": doc_type,
        "source": str(path),
        "chunks_indexed": len(chunks)
    }

def answer_from_memory():
    """
    Handles follow-up questions that modify the previous answer.

    Examples:
    - Can you elaborate on that?
    - Give me one pointer.
    - Summarize that in 2 bullets.
    - Make it simpler.

    This should NOT call FAISS.
    """

    if not chat_history:
        return "I do not have previous context yet."

    last_turn = chat_history[-1]
    last_question = last_turn["user"]
    last_answer = last_turn["ai"]

    memory_prompt = f"""
You are continuing a conversation.

The user is asking a follow-up about the previous answer.

Previous user question:
{last_question}

Previous AI answer:
{last_answer}

Current user follow-up:
{query}

Instructions:
- Answer only using the previous AI answer.
- Do not introduce a new topic.
- Do not search for new document information.
- If the user asks for one point, give only one point.
- If the user asks for a summary, summarize the previous answer.
- If the user asks to elaborate, expand only on the previous answer.
- Keep the response clear and relevant.

Answer:
"""

    response = pipe(
        memory_prompt,
        max_new_tokens=180,
        do_sample=False,
        truncation=True
    )

    return response[0]["generated_text"].strip()

def build_followup_rag_memory():
    """
    Builds a better RAG query for follow-up questions that need document retrieval.

    This does NOT ask the LLM to rewrite the question.
    It safely attaches the previous topic so retrieval stays focused.
    """

    if not chat_history:
        return query

    last_turn = chat_history[-1]

    last_question = last_turn["user"]
    last_answer = last_turn["ai"]

    followup_query = f"""
Previous topic/question:
{last_question}

Previous answer summary:
{last_answer[:600]}

Current follow-up question:
{query}

Task:
Answer the current follow-up question using the same topic and document context as the previous question.
"""

    return followup_query

def format_chat_history(history):
    """
    Converts previous conversation turns into a small text block.
    We keep it short because FLAN-T5 has a small input limit.
    """
    if not history:
        return "No previous conversation."

    recent_history = history[-MAX_MEMORY_TURNS:]

    formatted = ""
    for turn in recent_history:
        formatted += f"User: {turn['user']}\n"
        formatted += f"AI: {turn['ai']}\n"

    return formatted.strip()


def add_to_memory(user_question, ai_answer, source_documents=None):
    """
    Stores user question, AI answer, and optional source metadata.
    """

    sources = []

    if source_documents:
        for doc in source_documents:
            sources.append({
                "source": doc.metadata.get("source"),
                "title": doc.metadata.get("title"),
                "doc_type": doc.metadata.get("doc_type"),
                "file_name": doc.metadata.get("file_name"),
                "chunk_id": doc.metadata.get("chunk_id")
            })

    chat_history.append({
        "user": user_question,
        "ai": ai_answer,
        "sources": sources
    })

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Build route embeddings once
route_vectors = {}

for route, examples in ROUTE_EXAMPLES.items():
    vectors = embeddings.embed_documents(examples)
    route_vectors[route] = vectors


def route_question():
    """
    Classifies the user's question into:
    memory_transform, followup_retrieve, new_retrieve, or chat.
    """

    if not chat_history:
        return "new_retrieve"

    query_vector = embeddings.embed_query(query)

    scores = {}

    for route, vectors in route_vectors.items():
        similarities = [
            cosine_similarity(query_vector, vector)
            for vector in vectors
        ]
        scores[route] = max(similarities)

    best_route = max(scores, key=scores.get)

    print("\nRouter scores:")
    for route, score in scores.items():
        print(route, round(score, 3))

    print("Selected route:", best_route)

    return best_route

def rerank_documents(query, documents, top_n=3):
    """
    Reranks retrieved documents using a cross-encoder reranker.

    FAISS gives candidate chunks.
    The reranker scores each chunk against the query.
    We return the top_n most relevant chunks.
    """

    if not documents:
        return []

    pairs = []

    for doc in documents:
        pairs.append((query, doc.page_content))

    scores = reranker.predict(pairs)

    scored_docs = list(zip(documents, scores))

    scored_docs = sorted(
        scored_docs,
        key=lambda x: x[1],
        reverse=True
    )

    reranked_docs = [doc for doc, score in scored_docs[:top_n]]

    return reranked_docs

def retrieve_documents(query, doc_type_filter=None):
    """
    Retrieves candidate chunks from FAISS, optionally limited by document type.
    """

    search_kwargs = {
        "k": settings.retrieval_k,
        "fetch_k": settings.retrieval_fetch_k
    }

    if doc_type_filter:
        search_kwargs["filter"] = {
            "doc_type": doc_type_filter
        }

    try:
        return db.max_marginal_relevance_search(
            query,
            **search_kwargs
        )
    except Exception as error:
        raise VectorStoreError("Could not retrieve documents from the vector database.") from error


def format_api_conversation_history(conversation_history=None):
    if not conversation_history:
        return "No previous conversation."

    formatted_turns = []

    for turn in conversation_history:
        formatted_turns.append(f"User: {turn['question']}")
        formatted_turns.append(f"Assistant: {turn['answer']}")

    return "\n".join(formatted_turns)


def build_retrieval_query(query, conversation_history=None):
    if not conversation_history:
        return query

    recent_history = format_api_conversation_history(conversation_history)

    return f"""
Previous conversation:
{recent_history}

Current question:
{query}
"""


def build_final_messages(query, context, conversation_context):
    system_prompt = f"""
You are a careful document-grounded assistant.

Rules:
- Return only the final answer. Do not include reasoning, analysis, draft text, or self-corrections.
- Answer only using the document context.
- Use the conversation history only to understand what the current question refers to.
- If the document context does not contain the answer, respond exactly: {FALLBACK_ANSWER}
- If the document context does contain the answer, do not include the fallback sentence.
- Do not say "Wait", "I need to correct that", "Actually", or similar correction text.
- Prefer 3-5 concise bullet points when the context supports multiple points.
- Finish every sentence completely.
""".strip()

    user_prompt = f"""
Conversation history:
{conversation_context}

Document context:
{context}

Current question:
{query}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def render_generation_prompt(messages):
    tokenizer = getattr(pipe, "tokenizer", None)

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

    return "\n\n".join(
        f"{message['role'].title()}:\n{message['content']}"
        for message in messages
    ) + "\n\nAssistant:\n"


def clean_generated_answer(answer):
    cleaned_answer = answer.strip()

    for marker in SELF_CORRECTION_MARKERS:
        marker_index = cleaned_answer.find(marker)

        if marker_index != -1:
            cleaned_answer = cleaned_answer[:marker_index].strip()

    fallback_index = cleaned_answer.find(FALLBACK_ANSWER)

    if fallback_index > 0:
        cleaned_answer = cleaned_answer[:fallback_index].strip()

    return cleaned_answer or FALLBACK_ANSWER


def ask_rag(query, doc_type_filter=None, conversation_history=None):
    """
    Full RAG flow with reranking:
    1. Retrieve candidate chunks from FAISS
    2. Rerank chunks
    3. Build context
    4. Ask LLM
    """

    effective_conversation_history = select_conversation_history_for_query(
        query,
        conversation_history
    )

    retrieval_query = build_retrieval_query(
        query,
        conversation_history=effective_conversation_history
    )

    candidate_docs = retrieve_documents(
        retrieval_query,
        doc_type_filter=doc_type_filter
    )

    reranked_docs = rerank_documents(
        retrieval_query,
        documents=candidate_docs,
        top_n=settings.rerank_top_n
    )

    if not reranked_docs:
        return {
            "result": FALLBACK_ANSWER,
            "source_documents": []
        }

    context = "\n\n".join(
        doc.page_content for doc in reranked_docs
    ).strip()

    if not context:
        return {
            "result": FALLBACK_ANSWER,
            "source_documents": []
        }

    conversation_context = format_api_conversation_history(
        effective_conversation_history
    )
    final_prompt = render_generation_prompt(
        build_final_messages(
            query=query,
            context=context,
            conversation_context=conversation_context
        )
    )

    try:
        response = pipe(
            final_prompt,
            max_new_tokens=settings.llm_max_new_tokens,
            do_sample=False,
            truncation=True
        )
    except Exception as error:
        raise LLMGenerationError("The language model failed to generate an answer.") from error

    try:
        answer = clean_generated_answer(response[0]["generated_text"])
    except (IndexError, KeyError, TypeError) as error:
        raise LLMGenerationError("The language model returned an invalid response.") from error

    return {
        "result": answer,
        "source_documents": reranked_docs
    }
# -----------------------------
# 8. Chat loop
# -----------------------------
if __name__ == "__main__":
    print("Multi-document chatbot with memory ready!")
    print("Type 'exit' to quit.")
    print("Type 'memory' to view conversation memory.")
    print("Type 'clear' to clear memory.\n")


    while True:
        query = input("You: ")

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if query.lower() == "memory":
            print("\nConversation Memory:")
            print(format_chat_history(chat_history))
            print("\n" + "-" * 50 + "\n")
            continue

        if query.lower() == "clear":
            chat_history.clear()
            print("Memory cleared.\n")
            continue

        route = route_question()

        if route == "memory_transform":
            answer = answer_from_memory()

            print("\nAnswer:")
            print(answer)

            add_to_memory(query, answer)

        elif route == "followup_retrieve":
            question_for_rag = build_followup_rag_memory()

            print("\nQuestion sent to RAG:")
            print(question_for_rag)

            result = ask_rag(question_for_rag)

            answer = result["result"]

            print("\nAnswer:")
            print(answer)

            print("\nSources:")
            for doc in result["source_documents"]:
                metadata = doc.metadata
                print("-" * 40)
                print("Title:", metadata.get("title"))
                print("Type:", metadata.get("doc_type"))
                print("File:", metadata.get("file_name"))
                print("Chunk:", metadata.get("chunk_id"))
                print("Path:", metadata.get("source"))

            add_to_memory(query, answer, result["source_documents"])

        elif route == "new_retrieve":
            result = ask_rag(query)
            answer = result["result"]

            print("\nAnswer:")
            print(answer)

            print("\nSources:")
            for doc in result["source_documents"]:
                metadata = doc.metadata
                print("-" * 40)
                print("Title:", metadata.get("title"))
                print("Type:", metadata.get("doc_type"))
                print("File:", metadata.get("file_name"))
                print("Chunk:", metadata.get("chunk_id"))
                print("Path:", metadata.get("source"))

            add_to_memory(query, answer, result["source_documents"])

        else:
            answer = "Got it."

            print("\nAnswer:")
            print(answer)

            add_to_memory(query, answer)
