from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant answering questions about books.
Answer only from the context below. If the answer is not in the context, say you don't know.
When you use a passage, mention the book and chapter.

Context:
{context}"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

REWRITE_SYSTEM_PROMPT = """Given the conversation and a follow-up question, rewrite the follow-up
into a standalone question that can be understood without the conversation.
Do not answer it. If it is already standalone, return it unchanged."""

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITE_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)
