import logging
from operator import itemgetter
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableBranch, RunnablePassthrough

from app.rag.prompts import ANSWER_PROMPT, REWRITE_PROMPT

logger = logging.getLogger(__name__)


class RagResult(TypedDict):
    answer: str
    docs: list[Document]


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata['book']}, chapter {doc.metadata['chapter']}]\n{doc.page_content}"
        for doc in docs
    )


def log_standalone_question(inputs: dict[str, Any]) -> dict[str, Any]:
    if inputs["history"]:
        logger.info("Rewrote %r -> %r", inputs["question"], inputs["standalone"])
    return inputs


def create_rag_chain(
    retriever: BaseRetriever, llm: BaseChatModel
) -> Runnable[dict[str, Any], RagResult]:
    rewrite = REWRITE_PROMPT | llm | StrOutputParser()
    answer = ANSWER_PROMPT | llm | StrOutputParser()

    # First question of a session: nothing to rewrite, skip the extra LLM call.
    standalone_question = RunnableBranch(
        (lambda inputs: not inputs["history"], itemgetter("question")),
        rewrite,
    )

    return (
        RunnablePassthrough.assign(standalone=standalone_question)
        | log_standalone_question
        | RunnablePassthrough.assign(docs=itemgetter("standalone") | retriever)
        | RunnablePassthrough.assign(context=lambda inputs: format_docs(inputs["docs"]))
        | RunnablePassthrough.assign(answer=answer)
    ).pick(["answer", "docs"])
