"""Streamlit interface for the multilingual mini-RAG chatbot.

Run with:

    streamlit run app.py
"""

from __future__ import annotations

import html

import streamlit as st

from rag import (
    TOP_K,
    RagEngine,
    detect_language,
    get_api_key,
)

LANGUAGE_LABELS = {"en": "English", "fr": "Français", "ar": "العربية"}
MODE_EXTRACTIVE = "Extractive (TF-IDF only)"
MODE_GENERATIVE = "Generative (Gemini)"

EXAMPLES: dict[str, list[tuple[str, str]]] = {
    "en": [
        ("Tuition fees", "What do international students pay for each credit hour?"),
        ("Library hours", "What are the library hours on Saturday?"),
        ("Student housing", "How do I apply for student housing?"),
    ],
    "fr": [
        (
            "Frais de scolarité",
            "Combien coûte une heure de crédit pour les étudiants internationaux ?",
        ),
        (
            "Horaires de la bibliothèque",
            "Quels sont les horaires de la bibliothèque le samedi ?",
        ),
        ("Logement étudiant", "Comment demander un logement étudiant ?"),
    ],
    "ar": [
        ("الرسوم الدراسية", "كم يدفع الطالب الدولي عن كل ساعة معتمدة؟"),
        ("أوقات المكتبة", "ما هي أوقات عمل المكتبة يوم السبت؟"),
        ("السكن الجامعي", "كيف أتقدم بطلب للحصول على سكن جامعي؟"),
    ],
}

CSS = """
<style>
.block-container { max-width: 1100px; padding-top: 2rem; }
.answer-card {
    background: #f6f5ff;
    border-left: 4px solid #6c5ce7;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 0.35rem;
    color: #14142b;
    font-size: 1.03rem;
    line-height: 1.75;
}
.answer-card[dir="rtl"] {
    border-left: none;
    border-right: 4px solid #6c5ce7;
    line-height: 2.0;
}
.passage-card {
    background: #fbfbfd;
    border-left: 3px solid #d6d6e7;
    border-radius: 8px;
    padding: 10px 14px;
    color: #1c1c28;
    font-size: 0.95rem;
    line-height: 1.7;
}
.passage-card[dir="rtl"] {
    border-left: none;
    border-right: 3px solid #d6d6e7;
    line-height: 2.0;
}
[dir="rtl"] {
    font-family: "Segoe UI", Tahoma, "Noto Naskh Arabic", Arial, sans-serif;
    text-align: right;
}
.badge {
    display: inline-block;
    border-radius: 999px;
    padding: 0.22rem 0.7rem;
    margin-bottom: 0.35rem;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-ext { background: #e7f7ec; color: #146c2e; border: 1px solid #b7e4c7; }
.badge-gen { background: #efe9ff; color: #4b2fbf; border: 1px solid #d4c7ff; }
.badge-fallback { background: #fff4e5; color: #8a4b00; border: 1px solid #ffd8a8; }
</style>
"""


@st.cache_resource(show_spinner="Building the TF-IDF index…")
def load_engine() -> RagEngine:
    return RagEngine()


def render_text_block(text: str, language: str, css_class: str) -> None:
    """Render escaped user/document text in a styled block with correct bidi."""

    direction = "rtl" if language == "ar" else "ltr"
    safe_text = html.escape(text).replace("\n", "<br />")
    st.markdown(
        f'<div class="{css_class}" dir="{direction}" lang="{language}">'
        f"{safe_text}</div>",
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Nour University · Mini-RAG",
    page_icon="📚",
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)

engine = load_engine()
api_key = get_api_key()
has_key = api_key is not None

st.title("📚 Nour University · Mini-RAG")
st.caption(
    "Ask in **English**, **Français** or **العربية**. Answers are retrieved from "
    "a small FAQ corpus with TF-IDF + cosine similarity; Gemini can optionally "
    "turn the retrieved passages into a natural answer."
)

with st.sidebar:
    st.header("⚙️ Settings")
    mode_label = st.radio(
        "Answer mode",
        options=[MODE_EXTRACTIVE, MODE_GENERATIVE],
        index=0,
        disabled=not has_key,
        help=(
            "Extractive mode always works and returns the best passage. "
            "Generative mode needs a Gemini API key in .env."
        ),
    )
    if not has_key:
        st.caption("Generative mode is disabled until a key is configured.")

    if has_key:
        st.success("GEMINI_API_KEY detected")
    else:
        st.warning("No GEMINI_API_KEY found")
        st.caption(
            "Extractive mode is fully functional. To enable generative mode, "
            "copy `.env.example` to `.env` and add a key from Google AI Studio."
        )

    st.divider()
    st.subheader("Corpus")
    st.metric("Retrieval chunks", len(engine.chunks))
    st.metric("FAQ documents", len(engine.sources))
    st.caption("Languages: English · Français · العربية")
    st.caption(f"Top-k passages: {TOP_K}")

    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. FAQ files are split into Q&A chunks.\n"
        "2. TF-IDF vectors (word + character n-grams) index every chunk.\n"
        "3. The question is compared to all chunks with cosine similarity.\n"
        "4. The top 3 passages are shown; Gemini is optional."
    )
    st.caption("No model training. Everything runs on CPU.")

st.markdown("**💡 Try an example**")
example_columns = st.columns(3)
for column, (language, examples) in zip(example_columns, EXAMPLES.items()):
    with column:
        st.markdown(f"**{LANGUAGE_LABELS[language]}**")
        for index, (label, example_question) in enumerate(examples):
            if st.button(label, key=f"example_{language}_{index}"):
                st.session_state["question"] = example_question
                st.session_state.pop("result", None)
                st.rerun()

with st.form("question_form"):
    question = st.text_area(
        "Your question · Votre question · سؤالك",
        key="question",
        height=110,
        placeholder=(
            "e.g. What is the tuition per credit hour? / "
            "Combien coûte une heure de crédit ? / "
            "كم تبلغ تكلفة الساعة المعتمدة؟"
        ),
    )
    submitted = st.form_submit_button("Ask")

if submitted:
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        requested_mode = (
            "generative" if mode_label == MODE_GENERATIVE and has_key else "extractive"
        )
        spinner_text = (
            "Retrieving passages and asking Gemini…"
            if requested_mode == "generative"
            else "Retrieving the most relevant passages…"
        )
        try:
            with st.spinner(spinner_text):
                result = engine.answer(
                    question,
                    mode=requested_mode,
                    api_key=api_key,
                )
        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:  # noqa: BLE001 - surface a readable UI error
            st.error(f"Something went wrong: {exc}")
        else:
            st.session_state["result"] = result
            st.session_state["requested_generative"] = (
                requested_mode == "generative"
            )

result = st.session_state.get("result")
if result is None:
    st.info(
        "Ask a question above, or pick one of the examples. "
        "The answer will include the source passages and their similarity scores."
    )
else:
    st.divider()
    st.subheader("Answer")

    if result.mode == "generative":
        badge_class = "badge-gen"
        badge_text = f"✨ Generative · Gemini ({result.model})"
    elif st.session_state.get("requested_generative"):
        badge_class = "badge-fallback"
        badge_text = "⚠️ Extractive fallback · Gemini unavailable"
    else:
        badge_class = "badge-ext"
        badge_text = "🔍 Extractive · TF-IDF retrieval"

    st.markdown(
        f'<span class="badge {badge_class}">{badge_text}</span>',
        unsafe_allow_html=True,
    )

    if result.notice:
        st.warning(result.notice)

    answer_language = detect_language(result.answer)
    render_text_block(result.answer, answer_language, "answer-card")

    metric_columns = st.columns(3)
    metric_columns[0].metric("Top similarity", f"{result.top_score:.2f}")
    metric_columns[1].metric(
        "Question language",
        LANGUAGE_LABELS.get(result.language, result.language),
    )
    metric_columns[2].metric("Passages retrieved", len(result.passages))

    st.subheader("Source passages")
    st.caption(
        "Ranked by cosine similarity. The score is a lexical TF-IDF match, not "
        "a probability; the top passage is the extractive answer."
    )

    for rank, passage in enumerate(result.passages, start=1):
        with st.container(border=True):
            header_column, score_column = st.columns([4, 1])
            with header_column:
                st.markdown(
                    f"**{rank}. {passage.chunk.source}** · "
                    f"{LANGUAGE_LABELS.get(passage.chunk.language, passage.chunk.language)} · "
                    f"`{passage.chunk.chunk_id}`"
                )
            with score_column:
                st.caption(f"score {passage.score:.3f}")
            st.progress(min(max(passage.score, 0.0), 1.0))
            render_text_block(
                passage.chunk.text,
                passage.chunk.language,
                "passage-card",
            )

st.divider()
st.caption(
    "Retrieval: scikit-learn TF-IDF + cosine similarity · Generation: optional "
    "Gemini API · No model training or fine-tuning."
)
