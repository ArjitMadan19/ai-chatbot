from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from transformers import pipeline


# -----------------------------
# 1. Load embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# 2. Load FAISS vector database
# -----------------------------
db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)


# -----------------------------
# 3. Retriever
# -----------------------------
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10
    }
)


# -----------------------------
# 4. Load local Hugging Face LLM
# -----------------------------
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_new_tokens=512,
    truncation=True,
    do_sample=False
)

llm = HuggingFacePipeline(pipeline=pipe)


# -----------------------------
# 5. Prompt template
# -----------------------------
prompt_template = """
You are a legal document assistant.

Your job is to summarize the provided contract context in plain English.

Important rules:
- Do not answer with only a section number.
- Do not copy only headings.
- Use the actual clause text from the context.
- Give 3-5 clear bullet points.
- If the context contains a section heading, explain the content under that heading.
- If the answer is not in the context, say: I do not know based on the provided documents.

Context:
{context}

Question:
{question}

Plain-English answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# -----------------------------
# 6. Retrieval QA chain
# -----------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)


# -----------------------------
# 7. Conversation memory
# -----------------------------
chat_history = []
MAX_MEMORY_TURNS = 3


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


def add_to_memory(user_question, ai_answer):
    """
    Saves current question and answer into memory.
    """
    chat_history.append({
        "user": user_question,
        "ai": ai_answer
    })

def rewrite_followup_question(user_question, chat_history):
    """
    Rewrites follow-up questions into standalone questions using recent memory.
    This helps the retriever understand vague questions like:
    'What about that?', 'Who is responsible?', 'Summarize it.'
    """

    if not chat_history:
        return user_question

    recent_history = chat_history[-2:]

    history_text = ""
    for turn in recent_history:
        history_text += f"User: {turn['user']}\n"
        history_text += f"AI: {turn['ai'][:300]}\n"

    rewrite_prompt = f"""
Rewrite the current question as a standalone question.

Chat history:
{history_text}

Current question:
{user_question}

Standalone question:
"""

    rewritten = pipe(
        rewrite_prompt,
        max_new_tokens=80,
        truncation=True,
        do_sample=False
    )[0]["generated_text"].strip()

    if not rewritten:
        return user_question

    return rewritten
# -----------------------------
# 8. Chat loop
# -----------------------------
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

    standalone_question = rewrite_followup_question(query, chat_history)

    print("\nStandalone question used for retrieval:")
    print(standalone_question)

    result = qa.invoke({"query": standalone_question})

    answer = result["result"]

    print("\nAnswer:")
    print(answer)

    print("\nSources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source"))

    add_to_memory(query, answer)

    print("\n" + "-" * 50 + "\n")