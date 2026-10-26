import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.rag.embeddings import create_embeddings
from app.rag.retriever import create_retriever


class Evidence(BaseModel):
    book: str
    chapter: int
    quote: str


class Question(BaseModel):
    id: str
    type: Literal["fact", "paraphrase", "multi", "unanswerable"]
    question: str
    evidence: list[Evidence]
    reference: str


def load_questions(path: Path) -> list[Question]:
    questions: list[Question] = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                questions.append(Question.model_validate_json(line))
            except ValidationError as error:
                raise SystemExit(f"{path}:{number}: {error}") from error
    return questions


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def is_hit(doc: Document, evidence: Evidence) -> bool:
    return (
        doc.metadata["book"] == evidence.book
        and doc.metadata["chapter"] == evidence.chapter
        and normalize(evidence.quote) in normalize(doc.page_content)
    )


def first_hit_rank(docs: list[Document], evidence: list[Evidence]) -> int | None:
    for rank, doc in enumerate(docs, start=1):
        if any(is_hit(doc, item) for item in evidence):
            return rank
    return None


def count_found(docs: list[Document], evidence: list[Evidence]) -> int:
    return sum(any(is_hit(doc, item) for doc in docs) for item in evidence)


@dataclass
class Result:
    question: Question
    rank: int | None
    found: int
    top_score: float
    seconds: float


def top_score(docs: list[Document]) -> float:
    return docs[0].metadata["score"] if docs else 0.0


def warn_on_model_mismatch(model: str, collection: str) -> None:
    # book-parser puts the model name, made filename-safe, into the collection name.
    expected = re.sub(r"[^A-Za-z0-9_-]", "_", model.split("/")[-1])
    if expected not in collection:
        print(f"Warning: '{collection}' doesn't look like it was embedded with '{model}'")


def print_result(result: Result, docs: list[Document]) -> None:
    line = f"{result.question.id} rank {result.rank or 'miss'}"
    if result.question.type == "multi":
        line += f", evidence {result.found}/{len(result.question.evidence)}"
    print(line)
    if result.rank is None:
        for doc in docs[:3]:
            print(
                f"    {doc.metadata['book']} ch. {doc.metadata['chapter']} "
                f"score {doc.metadata['score']:.3f}: {doc.page_content[:80]}"
            )


def print_summary(results: list[Result], unanswerable_scores: list[float]) -> None:
    hits = sum(1 for result in results if result.rank is not None)
    mrr = sum(1 / result.rank for result in results if result.rank) / len(results)
    average_ms = 1000 * sum(result.seconds for result in results) / len(results)
    answerable_score = sum(result.top_score for result in results) / len(results)

    by_type: list[str] = []
    for kind in sorted({result.question.type for result in results}):
        of_kind = [result for result in results if result.question.type == kind]
        hits_of_kind = sum(1 for result in of_kind if result.rank is not None)
        by_type.append(f"{kind} {hits_of_kind}/{len(of_kind)}")

    print()
    print(f"Questions   {len(results)} with evidence, {len(unanswerable_scores)} without")
    print(f"Hit rate    {hits / len(results):.2f} ({hits}/{len(results)})")
    print(f"MRR         {mrr:.2f}")
    multi = [result for result in results if result.question.type == "multi"]
    if multi:
        found = sum(result.found / len(result.question.evidence) for result in multi) / len(multi)
        print(f"Multi       {found:.2f} of evidence found (n={len(multi)})")
    print(f"By type     {', '.join(by_type)}")
    # Answerable questions should score clearly higher; the gap shows whether
    # a score threshold could tell "nothing relevant" apart.
    top_scores = f"Top score   {answerable_score:.3f} with evidence"
    if unanswerable_scores:
        unanswerable_score = sum(unanswerable_scores) / len(unanswerable_scores)
        top_scores += f", {unanswerable_score:.3f} without"
    print(top_scores)
    print(f"Query time  {average_ms:.0f} ms average")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.retrieval")
    parser.add_argument("dataset", type=Path, help="questions in JSONL format")
    args = parser.parse_args()

    questions = load_questions(args.dataset)
    print(f"Loaded {len(questions)} questions")

    settings = get_settings()
    embeddings = create_embeddings(settings.ollama)
    retriever = create_retriever(settings.qdrant, embeddings, settings.retriever_k)
    print(f"Collection: {settings.qdrant.collection} (k={settings.retriever_k})")
    if settings.ollama.query_instruction:
        print(f"Query instruction: {settings.ollama.query_instruction}")
    warn_on_model_mismatch(settings.ollama.embedding_model, settings.qdrant.collection)

    # Loads the model in Ollama, so the first question isn't timed with it.
    embeddings.embed_query("warm-up")

    results: list[Result] = []
    unanswerable_scores: list[float] = []
    for question in questions:
        started = time.perf_counter()
        docs = retriever.invoke(question.question)
        seconds = time.perf_counter() - started
        if not question.evidence:
            unanswerable_scores.append(top_score(docs))
            print(f"{question.id} no evidence, top score {top_score(docs):.3f}")
            continue
        result = Result(
            question=question,
            rank=first_hit_rank(docs, question.evidence),
            found=count_found(docs, question.evidence),
            top_score=top_score(docs),
            seconds=seconds,
        )
        results.append(result)
        print_result(result, docs)

    if not results:
        raise SystemExit("No questions with evidence to evaluate")
    print_summary(results, unanswerable_scores)


if __name__ == "__main__":
    main()
