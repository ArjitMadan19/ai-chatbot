NEW_TOPIC_TERMS = {
    "attention",
    "bert",
    "contract",
    "agreement",
    "paper",
    "research",
    "document",
    "file",
    "pdf",
    "nda",
    "confidentiality"
}

FOLLOW_UP_WORDS = {
    "that",
    "this",
    "it",
    "they",
    "them",
    "those",
    "these",
    "above",
    "previous",
    "earlier",
    "same",
}

FOLLOW_UP_PHRASES = (
    "last answer",
    "previous answer",
    "tell me more",
    "explain more",
    "elaborate",
    "summarize that",
    "make it",
    "what about"
)


def normalize_question(question):
    return " ".join(question.lower().strip().split())


def has_new_topic_signal(normalized_question):
    question_terms = get_question_terms(normalized_question)
    return bool(question_terms.intersection(NEW_TOPIC_TERMS))


def has_follow_up_signal(normalized_question):
    question_terms = get_question_terms(normalized_question)

    if question_terms.intersection(FOLLOW_UP_WORDS):
        return True

    return any(phrase in normalized_question for phrase in FOLLOW_UP_PHRASES)


def get_question_terms(normalized_question):
    return set(
        normalized_question
        .replace("?", " ")
        .replace(".", " ")
        .replace(",", " ")
        .split()
    )


def is_follow_up_question(question, conversation_history=None):
    """
    Decides whether API memory should influence retrieval.

    A question like "summarize that" needs previous messages. A question like
    "what is in attention paper" is a new topic, so previous contract answers
    should not be embedded into the vector search query.
    """

    if not conversation_history:
        return False

    normalized_question = normalize_question(question)

    if not normalized_question:
        return False

    follow_up_signal = has_follow_up_signal(normalized_question)
    new_topic_signal = has_new_topic_signal(normalized_question)

    if new_topic_signal and not follow_up_signal:
        return False

    return follow_up_signal


def select_conversation_history_for_query(question, conversation_history=None):
    if is_follow_up_question(question, conversation_history):
        return conversation_history or []

    return []
