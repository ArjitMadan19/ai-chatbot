import json
import numpy as np
from sentence_transformers import SentenceTransformer

from app.rag.pipeline import ask_rag


# -----------------------------
# Load evaluation questions
# -----------------------------
with open("eval_questions.json", "r", encoding="utf-8") as file:
    eval_questions = json.load(file)


# -----------------------------
# Load embedding model for answer similarity
# -----------------------------
eval_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def answer_similarity(actual_answer, expected_answer):
    actual_embedding = eval_model.encode(actual_answer)
    expected_embedding = eval_model.encode(expected_answer)

    return cosine_similarity(actual_embedding, expected_embedding)


def source_matches(source_documents, expected_source_document):
    """
    Checks whether the retrieved source documents include the expected file.
    """

    for doc in source_documents:
        metadata = doc.metadata

        source = metadata.get("source", "")
        file_name = metadata.get("file_name", "")

        if expected_source_document in source or expected_source_document in file_name:
            return True

    return False


# -----------------------------
# Run evaluation
# -----------------------------
results = []

answer_pass_count = 0
source_pass_count = 0
overall_pass_count = 0

ANSWER_THRESHOLD = 0.55

for index, item in enumerate(eval_questions, start=1):
    question = item["question"]
    expected_answer = item["expected_answer"]
    expected_source = item["expected_source_document"]

    print(f"\nRunning question {index}: {question}")

    rag_result = ask_rag(question)

    actual_answer = rag_result["result"]
    source_documents = rag_result["source_documents"]

    similarity_score = answer_similarity(actual_answer, expected_answer)
    source_pass = source_matches(source_documents, expected_source)

    answer_pass = bool(similarity_score >= ANSWER_THRESHOLD)
    source_pass = bool(source_pass)
    overall_pass = bool(answer_pass and source_pass)

    if answer_pass:
        answer_pass_count += 1

    if source_pass:
        source_pass_count += 1

    if overall_pass:
        overall_pass_count += 1

    result_record = {
        "question": question,
        "expected_answer": expected_answer,
        "actual_answer": actual_answer,
        "expected_source_document": expected_source,
        "answer_similarity_score": round(float(similarity_score), 3),
        "answer_pass": answer_pass,
        "source_pass": source_pass,
        "overall_pass": overall_pass
    }

    results.append(result_record)

    print("Answer similarity:", round(float(similarity_score), 3))
    print("Answer pass:", answer_pass)
    print("Source pass:", source_pass)
    print("Overall pass:", overall_pass)


# -----------------------------
# Print final score
# -----------------------------
total = len(eval_questions)

print("\n==============================")
print("Evaluation Summary")
print("==============================")
print(f"Total questions: {total}")
print(f"Answer pass: {answer_pass_count}/{total}")
print(f"Source pass: {source_pass_count}/{total}")
print(f"Overall pass: {overall_pass_count}/{total}")


# -----------------------------
# Save detailed results
# -----------------------------
with open("eval_results.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print("\nDetailed results saved to eval_results.json")
