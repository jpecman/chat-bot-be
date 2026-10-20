from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough

from app.rag.prompts import ANSWER_PROMPT


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata['book']}, chapter {doc.metadata['chapter']}]\n{doc.page_content}"
        for doc in docs
    )


def create_rag_chain(retriever: BaseRetriever, llm: BaseChatModel) -> Runnable[str, str]:
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )
