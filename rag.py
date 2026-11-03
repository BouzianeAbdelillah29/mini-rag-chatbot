"""Core retrieval and answering logic for the multilingual mini-RAG chatbot.

Pipeline
--------
1. Load the three FAQ files (English, French, Arabic) from ``data/``.
2. Split every file into self-contained question/answer chunks.
3. Fit a TF-IDF index on the chunks: word 1-2 grams plus character 3-5 grams.
4. For a question, compute cosine similarity against every chunk and return
   the top-k passages.
5. Optionally ask Gemini to write a grounded answer from those passages.

No model is trained or fine-tuned. Retrieval runs on CPU and always works;
generation is optional and needs ``GEMINI_API_KEY`` in a local ``.env`` file.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

LANGUAGE_FILES = {
    "en": "faqs_en.txt",
    "fr": "faqs_fr.txt",
    "ar": "faqs_ar.txt",
}
LANGUAGE_NAMES = {"en": "English", "fr": "French", "ar": "Arabic"}

TOP_K = 3
MAX_CHUNK_CHARS = 700
WORD_WEIGHT = 1.0
CHAR_WEIGHT = 0.45
LOW_CONFIDENCE_THRESHOLD = 0.08

# Tried in order. If the first model name is rejected, the second is used.
GEMINI_MODELS = ("gemini-2.5-flash", "gemini-3.8-flash")

_PLACEHOLDER_KEYS = {
    "",
    "your_key_here",
    "your-key-here",
    "your_api_key_here",
    "replace_me",
}

_ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]"
)
_ARABIC_CHARS = re.compile(r"[\u0600-\u06FF]")
_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_UNDERSCORES = re.compile(r"_+")

_FRENCH_HINTS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "est", "quel",
    "quelle", "quels", "quelles", "combien", "comment", "ou", "quand",
    "pour", "dans", "avec", "je", "mon", "ma", "mes", "au", "aux", "et",
    "en", "sur", "par", "que", "qui", "ne", "pas", "faut", "vous", "nous",
    "puis", "etudiant", "etudiants", "cours", "universite",
}
_ENGLISH_HINTS = {
    "the", "is", "are", "what", "how", "when", "where", "which", "do",
    "does", "can", "i", "my", "for", "of", "to", "in", "on", "and", "a",
    "an", "much", "many", "cost", "need", "there", "student", "students",
    "university", "course", "courses", "fee", "fees",
}


class GenerationError(RuntimeError):
    """Raised when Gemini cannot produce an answer with any configured model."""


@dataclass(frozen=True)
class Chunk:
    """A single retrievable FAQ passage."""

    chunk_id: str
    source: str
    language: str
    text: str


@dataclass(frozen=True)
class RetrievedPassage:
    """A chunk together with its cosine similarity to the question."""

    chunk: Chunk
    score: float


@dataclass
class AnswerResult:
    """Everything the UI needs to render one answer."""

    question: str
    language: str
    mode: str  # "extractive" or "generative"
    answer: str
    passages: list[RetrievedPassage]
    top_score: float
    model: str | None = None
    notice: str | None = None


def normalize_text(text: str) -> str:
    """Normalize text for matching without changing the displayed passages.

    Lowercasing, accent removal and Arabic letter/diacritic normalization make
    retrieval tolerant of spelling variants (e.g. ``étudiants`` / ``etudiants``,
    ``أ`` / ``ا``) while the original text is always shown to the user.
    """

    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _ARABIC_DIACRITICS.sub("", text)
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )
    # Strip Latin combining accents (é -> e) and any remaining marks.
    text = "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    text = _NON_WORD.sub(" ", text)
    text = _UNDERSCORES.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def detect_language(text: str) -> str:
    """Return ``en``, ``fr`` or ``ar`` for a question or passage."""

    if len(_ARABIC_CHARS.findall(text)) >= 2:
        return "ar"
    words = set(re.findall(r"[a-z\u00e0-\u00ff']+", text.lower()))
    french_hits = len(words & _FRENCH_HINTS)
    english_hits = len(words & _ENGLISH_HINTS)
    return "fr" if french_hits > english_hits else "en"


def get_api_key(env_file: str | Path | None = None) -> str | None:
    """Load ``GEMINI_API_KEY`` from the environment or the project ``.env``.

    Returns ``None`` when the variable is missing or still contains the
    placeholder from ``.env.example``. The key is never printed or logged.
    """

    path = Path(env_file) if env_file else PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=path, override=False)
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if key.lower() in _PLACEHOLDER_KEYS or key.lower().startswith("your_"):
        return None
    return key or None


def _split_long_block(block: str, max_chars: int) -> list[str]:
    """Split an oversized block on sentence boundaries, keeping it readable."""

    if len(block) <= max_chars:
        return [block]

    sentences = re.split(r"(?<=[.!?])\s+", block)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces or [block]


def load_chunks(data_dir: str | Path | None = None) -> list[Chunk]:
    """Read the FAQ files and return one chunk per question/answer block."""

    directory = Path(data_dir) if data_dir else DATA_DIR
    chunks: list[Chunk] = []

    for language, filename in LANGUAGE_FILES.items():
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(
                f"FAQ file not found: {path}. Expected data/{filename}."
            )

        raw = path.read_text(encoding="utf-8")
        blocks = [
            block.strip()
            for block in re.split(r"\n\s*\n", raw.strip())
            if block.strip()
        ]
        file_chunks: list[Chunk] = []
        for block in blocks:
            lines = [
                line
                for line in block.splitlines()
                if not line.lstrip().startswith("#")
            ]
            clean = "\n".join(lines).strip()
            if not clean:
                continue
            for piece in _split_long_block(clean, MAX_CHUNK_CHARS):
                file_chunks.append(
                    Chunk(
                        chunk_id=f"{language}-{len(file_chunks) + 1:02d}",
                        source=filename,
                        language=language,
                        text=piece,
                    )
                )
        chunks.extend(file_chunks)

    if not chunks:
        raise ValueError(f"No FAQ chunks were loaded from {directory}.")
    return chunks


class RagEngine:
    """TF-IDF retriever over the multilingual FAQ corpus."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.chunks: list[Chunk] = load_chunks(self.data_dir)

        texts = [chunk.text for chunk in self.chunks]

        self._word_vectorizer = TfidfVectorizer(
            preprocessor=normalize_text,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self._char_vectorizer = TfidfVectorizer(
            preprocessor=normalize_text,
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=1,
        )

        word_matrix = self._word_vectorizer.fit_transform(texts) * WORD_WEIGHT
        char_matrix = self._char_vectorizer.fit_transform(texts) * CHAR_WEIGHT
        self._matrix = hstack([word_matrix, char_matrix]).tocsr()

    @property
    def sources(self) -> list[str]:
        """Unique document names, in load order."""

        return list(dict.fromkeys(chunk.source for chunk in self.chunks))

    def retrieve(
        self, question: str, k: int = TOP_K
    ) -> list[RetrievedPassage]:
        """Return the ``k`` passages with the highest cosine similarity."""

        question = (question or "").strip()
        if not question:
            raise ValueError("Please enter a question.")

        word_query = self._word_vectorizer.transform([question])
        char_query = self._char_vectorizer.transform([question])
        query = hstack([word_query * WORD_WEIGHT, char_query * CHAR_WEIGHT]).tocsr()

        scores = cosine_similarity(query, self._matrix).ravel()
        limit = max(1, min(int(k), len(self.chunks)))
        best_indices = scores.argsort()[::-1][:limit]

        return [
            RetrievedPassage(
                chunk=self.chunks[index],
                score=round(float(scores[index]), 4),
            )
            for index in best_indices
        ]

    def answer(
        self,
        question: str,
        mode: str = "extractive",
        api_key: str | None = None,
        k: int = TOP_K,
    ) -> AnswerResult:
        """Answer in extractive mode or, when possible, with Gemini.

        Generation failures never break the app: the result falls back to the
        best retrieved passage and explains what happened in ``notice``.
        """

        if mode not in {"extractive", "generative"}:
            raise ValueError("mode must be 'extractive' or 'generative'")

        question = (question or "").strip()
        if not question:
            raise ValueError("Please enter a question.")

        language = detect_language(question)
        passages = self.retrieve(question, k=k)
        if not passages:
            return AnswerResult(
                question=question,
                language=language,
                mode="extractive",
                answer="No passages were found in the FAQ corpus.",
                passages=[],
                top_score=0.0,
            )

        best = passages[0]
        low_confidence_notice = None
        if best.score < LOW_CONFIDENCE_THRESHOLD:
            low_confidence_notice = (
                "Low similarity score: the corpus may not contain a clear "
                "answer to this question."
            )

        if mode == "generative":
            if not api_key:
                return AnswerResult(
                    question=question,
                    language=language,
                    mode="extractive",
                    answer=best.chunk.text,
                    passages=passages,
                    top_score=best.score,
                    notice=(
                        "Generative mode is unavailable because no "
                        "GEMINI_API_KEY was found. Showing the best retrieved "
                        "passage instead."
                    ),
                )
            try:
                generated, model = generate_answer(question, passages, api_key)
                return AnswerResult(
                    question=question,
                    language=language,
                    mode="generative",
                    answer=generated,
                    passages=passages,
                    top_score=best.score,
                    model=model,
                )
            except GenerationError as exc:
                return AnswerResult(
                    question=question,
                    language=language,
                    mode="extractive",
                    answer=best.chunk.text,
                    passages=passages,
                    top_score=best.score,
                    notice=(
                        "Gemini generation failed, so the best retrieved "
                        f"passage is shown instead. Details: {exc}"
                    ),
                )

        return AnswerResult(
            question=question,
            language=language,
            mode="extractive",
            answer=best.chunk.text,
            passages=passages,
            top_score=best.score,
            notice=low_confidence_notice,
        )


def _format_context(passages: Sequence[RetrievedPassage]) -> str:
    lines = []
    for index, passage in enumerate(passages, start=1):
        language = LANGUAGE_NAMES.get(passage.chunk.language, passage.chunk.language)
        lines.append(
            f"[Passage {index} | {passage.chunk.source} | {language} | "
            f"similarity {passage.score:.3f}]\n{passage.chunk.text}"
        )
    return "\n\n".join(lines)


def generate_answer(
    question: str,
    passages: Sequence[RetrievedPassage],
    api_key: str,
    models: Sequence[str] = GEMINI_MODELS,
) -> tuple[str, str]:
    """Generate a grounded answer with Gemini, with a model fallback.

    The prompt restricts the model to the retrieved passages and asks it to
    answer in the language of the question. Returns ``(answer, model_used)``.
    """

    if not passages:
        raise GenerationError("No passages were retrieved.")
    if not api_key:
        raise GenerationError("No Gemini API key was provided.")

    language = detect_language(question)
    prompt = (
        "You are the virtual assistant of Nour University, a fictional "
        "university used in a portfolio demo.\n"
        "Answer the question using ONLY the context passages below. "
        "Do not invent fees, dates, names or policies. If the context does "
        "not contain the answer, say so briefly and suggest contacting the "
        "relevant office.\n"
        f"Answer in {LANGUAGE_NAMES[language]} and keep the answer under "
        "120 words. Be direct, friendly and factual.\n\n"
        f"Context:\n{_format_context(passages)}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise GenerationError(
            "google-genai is not installed; run pip install -r requirements.txt"
        ) from exc

    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:  # pragma: no cover - SDK specific
        raise GenerationError(f"Could not create the Gemini client: {exc}") from exc

    errors: list[str] = []
    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = (getattr(response, "text", None) or "").strip()
            if text:
                return text, model
            errors.append(f"{model}: empty response")
        except Exception as exc:  # network, quota, invalid model name, ...
            message = _WHITESPACE.sub(" ", str(exc)).strip()[:180]
            errors.append(f"{model}: {type(exc).__name__}: {message}")

    raise GenerationError("; ".join(errors))
