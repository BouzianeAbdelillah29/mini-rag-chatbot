"""Evaluation of the TF-IDF retriever on 9 held-out questions.

Run from the project folder:

    python eval.py

For every question (3 per language) the script checks whether the passage
containing the expected answer appears in the top-1 and top-3 results.
No generation is involved: this measures retrieval quality only.
"""

from __future__ import annotations

import re
import sys
import unicodedata

from rag import RagEngine

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# (language, question, expected answer fragment, topic)
TEST_CASES = [
    (
        "en",
        "What do international students pay for each credit hour?",
        "420 USD",
        "Tuition fees",
    ),
    (
        "en",
        "What are the library hours on Saturday?",
        "Saturday from 9:00 to 18:00",
        "Library hours",
    ),
    (
        "en",
        "How many internship hours does a bachelor's student need?",
        "120 hours",
        "Internship requirement",
    ),
    (
        "fr",
        "Combien paient les étudiants internationaux par heure de crédit ?",
        "420 USD",
        "Frais de scolarité",
    ),
    (
        "fr",
        "Quels sont les horaires de la bibliothèque le samedi ?",
        "samedi de 9:00 à 18:00",
        "Horaires de la bibliothèque",
    ),
    (
        "fr",
        "Combien d'heures de stage un étudiant en licence doit-il effectuer ?",
        "120 heures",
        "Stage obligatoire",
    ),
    (
        "ar",
        "كم تبلغ تكلفة الساعة المعتمدة للطالب الدولي؟",
        "420 دولارًا",
        "الرسوم الدراسية",
    ),
    (
        "ar",
        "ما هي أوقات عمل المكتبة يوم السبت؟",
        "السبت من 9:00 إلى 18:00",
        "أوقات المكتبة",
    ),
    (
        "ar",
        "كم عدد ساعات التدريب المطلوبة لطلاب البكالوريوس؟",
        "120 ساعة",
        "التدريب الإلزامي",
    ),
]

LANGUAGE_LABELS = {"en": "EN", "fr": "FR", "ar": "AR"}


def _match_key(text: str) -> str:
    """Case-, accent- and whitespace-insensitive key for substring checks."""

    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def _contains(haystack: str, needle: str) -> bool:
    return _match_key(needle) in _match_key(haystack)


def main() -> int:
    engine = RagEngine()
    total = len(TEST_CASES)
    top1_hits = 0
    top3_hits = 0
    per_language: dict[str, list[int]] = {"en": [0, 0], "fr": [0, 0], "ar": [0, 0]}

    print("Multilingual mini-RAG - retrieval evaluation")
    print("=" * 78)
    print(
        f"Corpus: {len(engine.chunks)} chunks from "
        f"{len(engine.sources)} documents ({', '.join(engine.sources)})"
    )
    print(f"Metric: top-k cosine similarity over TF-IDF (word + char n-grams), k=3")
    print("=" * 78)

    for index, (language, question, expected, topic) in enumerate(TEST_CASES, start=1):
        passages = engine.retrieve(question, k=3)
        top1_ok = bool(passages) and _contains(passages[0].chunk.text, expected)
        top3_ok = any(_contains(passage.chunk.text, expected) for passage in passages)

        top1_hits += int(top1_ok)
        top3_hits += int(top3_ok)
        per_language[language][0] += int(top1_ok)
        per_language[language][1] += int(top3_ok)

        top1_flag = "OK  " if top1_ok else "MISS"
        top3_flag = "OK  " if top3_ok else "MISS"
        print(
            f"\n[{index}/{total}] {LANGUAGE_LABELS[language]} | {topic}\n"
            f"  Q: {question}\n"
            f"  expected: {expected!r}\n"
            f"  top-1: {top1_flag}  score={passages[0].score:.3f}  "
            f"chunk={passages[0].chunk.chunk_id} ({passages[0].chunk.source})\n"
            f"  top-3: {top3_flag}  score={passages[0].score:.3f}\n"
            f"  retrieved: "
            + " | ".join(
                f"{passage.chunk.chunk_id} {passage.score:.3f}"
                for passage in passages
            )
        )

    def percentage(hits: int) -> str:
        return f"{hits / total * 100:.1f}%"

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"Top-1 retrieval accuracy: {top1_hits}/{total} ({percentage(top1_hits)})")
    print(f"Top-3 retrieval accuracy: {top3_hits}/{total} ({percentage(top3_hits)})")
    for language, (language_top1, language_top3) in per_language.items():
        print(
            f"  {LANGUAGE_LABELS[language]}: "
            f"top-1 {language_top1}/3, top-3 {language_top3}/3"
        )

    if top3_hits < total:
        print(
            "\nSome expected answers were not retrieved. Check the source "
            "documents and the question wording before tuning the retriever."
        )
    else:
        print("\nAll expected answers were retrieved within the top 3 passages.")

    return 0 if top3_hits == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
