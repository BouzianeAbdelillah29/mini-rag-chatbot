# Nour University · Multilingual Mini-RAG Chatbot

A small, fully local **retrieval-augmented generation (RAG)** chatbot that answers
questions about a fictional university in **English, French and Arabic**.

The project is deliberately simple to show the *retrieval* part clearly: FAQ
documents are split into passages, indexed with **TF-IDF**, and ranked with
**cosine similarity**. An extractive mode always works and returns the best
passage plus its score. A generative mode can optionally use the **Google Gemini
API** to turn the retrieved passages into a natural answer. **No model is trained,
fine-tuned or downloaded, and no GPU is needed.**

## Features

- **3 languages** — English, French and Arabic questions over a parallel FAQ corpus.
- **Extractive answers (always available)** — the best-matching question/answer
  passage with its cosine similarity score and the top-3 sources.
- **Generative answers (optional)** — Gemini writes an answer grounded only in
  the retrieved passages; if the key or API is unavailable, the app falls back
  to extractive mode automatically.
- **Arabic-aware preprocessing and display** — diacritic/letter normalization for
  matching, and right-to-left HTML rendering with an Arabic-capable font stack.
- **Evaluation included** — 9 held-out questions (3 per language) measure top-1
  and top-3 retrieval accuracy.
- **CPU-only and reproducible** — pinned dependencies, no training, no vector
  database, no large model downloads.

## How mini-RAG works

### Indexing (once, at app startup)

1. **Load documents** — `rag.py` reads the three UTF-8 FAQ files from `data/`.
2. **Split into chunks** — every block separated by a blank line is one
   self-contained question/answer chunk. Each file has 17 chunks, 51 in total.
3. **Normalize for matching** — lowercase; remove Latin accents
   (`étudiants` → `etudiants`) and Arabic diacritics; unify Arabic letter forms
   (`أ`/`إ`/`آ` → `ا`, `ة` → `ه`, `ى` → `ي`). Displayed passages are **not**
   modified.
4. **Vectorize with TF-IDF** — two sparse matrices are concatenated:
   - word 1–2 grams, which capture exact terms and short phrases;
   - character 3–5 grams, which make retrieval robust to spelling variants,
     accents, prefixes and Arabic morphology.
5. **Store the index** — the fitted vectorizers and the chunk matrix live in
   memory for the lifetime of the app (cached by Streamlit). TF-IDF is an
   unsupervised counting/weighting step, **not** model training.

### Answering a question

1. The question is normalized with the same function as the documents.
2. It is vectorized and compared to all 51 chunks using **cosine similarity**.
3. The **top 3** passages are returned with their scores.
4. **Extractive mode:** the top-1 chunk is the answer. Nothing can fail here.
5. **Generative mode (optional):** the top passages are inserted into a prompt
   that tells Gemini to answer only from that context, in the question's
   language. The model is asked not to invent dates, fees or policies. If the
   request fails, the app displays the extractive passage and a warning.

```text
data/faqs_{en,fr,ar}.txt
        │  load + split into Q&A chunks
        ▼
normalize_text()  (case, accents, Arabic diacritics/letters)
        ▼
TF-IDF: word(1,2)  ⊕  char_wb(3,5)
        ▼
cosine_similarity(question, chunks) ──► top 3 passages
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
        Extractive: best passage + score              Generative: Gemini prompt
        (always available)                            (optional, grounded)
```

### Why TF-IDF?

For a small FAQ corpus, TF-IDF is fast, deterministic, transparent and requires
no downloads. It also keeps the project's focus on the RAG architecture rather
than on model plumbing. Its limitation is that it is **lexical**: it cannot
match a question to a passage that uses completely different words. The
character n-grams soften this, but true cross-lingual semantic search would
need multilingual embeddings (see *Limitations and next steps*).

## Project structure

```text
D:\mini-rag-chatbot\
├── app.py              Streamlit interface
├── rag.py              document loading, chunking, TF-IDF, retrieval, Gemini
├── eval.py             9-question retrieval evaluation
├── data\
│   ├── faqs_en.txt     17 English Q&A passages
│   ├── faqs_fr.txt     17 French Q&A passages
│   └── faqs_ar.txt     17 Arabic Q&A passages
├── requirements.txt    pinned dependencies
├── .env.example        template for the optional Gemini key
├── .gitignore          ignores .env, __pycache__, Streamlit cache
├── README.md
└── .venv\              local virtual environment (ignored, not in git)
```

## Installation (Windows, PowerShell)

Python 3.14 was used for development. All commands assume the project is at
`D:\mini-rag-chatbot`.

```powershell
cd "D:\mini-rag-chatbot"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, use the interpreter directly:

```powershell
& "D:\mini-rag-chatbot\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## Running the app

```powershell
cd "D:\mini-rag-chatbot"
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Or without activating the environment:

```powershell
& "D:\mini-rag-chatbot\.venv\Scripts\streamlit.exe" run "D:\mini-rag-chatbot\app.py"
```

Streamlit opens `http://localhost:8501`. Type a question in English, French or
Arabic and press **Ask**. The page shows:

- a badge indicating **Extractive · TF-IDF retrieval**, **Generative · Gemini**,
  or **Extractive fallback**;
- the answer (right-to-left for Arabic);
- the top-3 source passages with chunk IDs, languages and similarity scores.

### Optional: enable generative mode

1. Create a free API key in [Google AI Studio](https://aistudio.google.com/app/apikey).
2. In the project folder, copy the template and open it:

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

3. Replace `your_key_here` with your key and save:

   ```dotenv
   GEMINI_API_KEY=your_real_key_here
   ```

4. Restart Streamlit. The sidebar should say **GEMINI_API_KEY detected** and the
   **Generative (Gemini)** option becomes selectable.

The key is read only from `.env` (via `python-dotenv`) or the process
environment. It is never hardcoded, printed or committed. `.env` is excluded by
`.gitignore`. Without a key the app remains fully functional in extractive mode
and clearly states that generative mode is unavailable.

The generation call uses `gemini-2.5-flash` first. If that model name returns an
error, it retries with `gemini-3.8-flash`; if both fail, the app falls back to the
extractive answer with a visible notice.

## Example questions

**English**

- *What are the library hours on Saturday?* → 9:00–18:00
- *How do I apply for student housing?* → submit the form by 30 June
- *Does the university offer psychological support?* → free, confidential
  counseling; up to four sessions per semester

**Français**

- *Quelles bourses l'Université Nour propose-t-elle ?* → bourse d'excellence,
  30 % des frais de scolarité
- *Quand commence l'année universitaire ?* → le 15 septembre
- *Puis-je étudier à temps partiel ?* → oui, 6 crédits maximum par semestre

**العربية**

- *هل توجد نوادٍ طلابية؟* → 24 ناديًا طلابيًا
- *ما هي المنح الدراسية التي تقدمها جامعة نور؟* → منحة التفوق تغطي 30% من الرسوم
- *كيف أتواصل مع مكتب القبول؟* → admissions@nour-university.edu أو الهاتف

Questions do not need to copy the wording of the FAQ. For example, the FAQ asks
*"How much does one credit hour cost for international students?"* while the
evaluation asks *"What do international students pay for each credit hour?"*.
The retrieval still ranks the correct passage first (score 0.633).

## Evaluation

Run:

```powershell
cd "D:\mini-rag-chatbot"
& "D:\mini-rag-chatbot\.venv\Scripts\python.exe" eval.py
```

`eval.py` contains 9 held-out questions (3 per language). For each one it checks
whether the passage containing the expected answer appears in the top-1 and
top-3 retrieved chunks. It measures **retrieval only** — no generation and no
API key are involved. The threshold for top-3 is intentionally the same as the
app's retrieval depth.

### Results (actual run, 51 chunks)

| # | Lang | Question | Top-1 | Top-3 | Top-1 score |
|---|------|----------|:-----:|:-----:|------------:|
| 1 | EN | What do international students pay for each credit hour? | ✅ | ✅ | 0.633 |
| 2 | EN | What are the library hours on Saturday? | ✅ | ✅ | 0.456 |
| 3 | EN | How many internship hours does a bachelor's student need? | ✅ | ✅ | 0.390 |
| 4 | FR | Combien paient les étudiants internationaux par heure de crédit ? | ✅ | ✅ | 0.792 |
| 5 | FR | Quels sont les horaires de la bibliothèque le samedi ? | ✅ | ✅ | 0.530 |
| 6 | FR | Combien d'heures de stage un étudiant en licence doit-il effectuer ? | ✅ | ✅ | 0.507 |
| 7 | AR | كم تبلغ تكلفة الساعة المعتمدة للطالب الدولي؟ | ✅ | ✅ | 0.292 |
| 8 | AR | ما هي أوقات عمل المكتبة يوم السبت؟ | ✅ | ✅ | 0.468 |
| 9 | AR | كم عدد ساعات التدريب المطلوبة لطلاب البكالوريوس؟ | ✅ | ✅ | 0.318 |

```text
Top-1 retrieval accuracy: 9/9 (100.0%)
Top-3 retrieval accuracy: 9/9 (100.0%)
  EN: top-1 3/3, top-3 3/3
  FR: top-1 3/3, top-3 3/3
  AR: top-1 3/3, top-3 3/3
```

The score is cosine similarity, not a probability. It depends on question length
and wording, so compare scores within one query rather than across queries.

## Testing notes

- `python eval.py` completes with exit code 0 when all expected answers are in
  the top 3.
- The Streamlit app was exercised with `streamlit.testing.v1.AppTest` for one
  question in each language; the expected answer text and the Arabic
  `dir="rtl"` rendering were present, and no exceptions were raised.
- The Gemini model fallback and the extractive fallback path were tested with a
  stubbed SDK — no real API key is needed for those checks.
- Without `.env`, the sidebar reports **No GEMINI_API_KEY found**, generative
  mode is disabled, and every test question is still answered extractively.

## Limitations and next steps

- **Lexical retrieval.** TF-IDF cannot bridge synonyms or fully different
  phrasings, and cross-lingual matching is limited (the parallel corpus helps,
  but there is no shared semantic space). The evaluation questions are
  paraphrases, not arbitrary natural-language questions.
- **Tiny corpus.** 51 chunks make retrieval fast and easy to inspect, but the
  scores are corpus-specific and should not be read as benchmark results.
- **Generation depends on the API.** Quota, network or model-name errors are
  handled by falling back to extractive mode; they never leave the user without
  an answer.
- **Fictional data.** Nour University, its fees, dates and contacts are invented
  for the demo and must not be used as real policy.
- **Re-indexing.** The index is rebuilt from the files at startup; adding
  documents just means adding chunks and restarting the app.

Natural next steps for a larger version: multilingual sentence embeddings and a
vector index, hybrid lexical + semantic retrieval, a reranker, conversation
memory, streaming answers, and ingestion of PDFs. Those would add model
downloads and complexity; this project intentionally stops at transparent
TF-IDF retrieval plus optional grounded generation.
