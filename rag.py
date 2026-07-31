"""
rag.py — A small Retrieval-Augmented Generation (RAG) layer for the
Fashion Identity Classifier.

Why this exists
----------------
Previously, the app's recommendations were a handful of static, hardcoded
strings per segment (see the old SHOPPING_SEGMENTS / results.html if/elif
blocks). That doesn't scale: adding nuance means editing template code, and
every user in a segment sees identical advice regardless of *how* they
answered.

This module implements the standard three RAG steps, kept intentionally
simple and dependency-light (it reuses scikit-learn, which the project
already depends on — no new packages required):

  1. Retrieve  — TF-IDF + cosine similarity over a small knowledge base of
                 style/shopping-advice documents (data/knowledge_base.json).
  2. Augment   — the retrieved chunks are assembled into a context block.
  3. Generate  — a lightweight composer turns the retrieved context into a
                 grounded, cited recommendation. If ANTHROPIC_API_KEY is
                 present in the environment, this step calls the Claude API
                 to produce a fluent, natural-language synthesis of the
                 retrieved context; otherwise it falls back to a clean
                 extractive template (so the app fully works offline).

This keeps every fact in the final output traceable back to a specific
knowledge-base entry, which is the core value of RAG over asking a language
model to answer purely from memory.
"""

import os
import json
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Resolve the default knowledge-base path relative to this file's own
# location rather than the current working directory, so retrieval works
# the same regardless of where the app is launched from.
_DEFAULT_KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'knowledge_base.json')


class FashionRAG:
    def __init__(self, kb_path: str = None):
        self.kb_path = kb_path or _DEFAULT_KB_PATH
        self.documents: List[Dict] = []
        self.vectorizer = None
        self.doc_matrix = None
        self._load_and_index()

    def _load_and_index(self):
        """Load the knowledge base and build the TF-IDF index (the
        'vector store' in this basic RAG setup)."""
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(
                f"Knowledge base not found at '{self.kb_path}'. Make sure "
                "data/knowledge_base.json exists, or pass an explicit kb_path."
            )

        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        if not self.documents:
            raise ValueError(f"Knowledge base at '{self.kb_path}' is empty.")

        corpus = [doc["text"] for doc in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 4, segment: str = None) -> List[Dict]:
        """Step 1: Retrieve the most relevant knowledge-base chunks for a
        query. When a segment is given, candidates are restricted to that
        segment's own documents plus the segment-agnostic 'General' ones —
        a hard filter rather than a small score boost, since a soft boost
        can still let a strongly-matching off-segment document (e.g.
        'Sustainable Conscious' style advice) outrank correct, on-segment
        advice. Only falls back to the full corpus if that leaves too few
        candidates to fill top_k."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix)[0]

        scored = list(zip(scores, self.documents))

        if segment:
            in_scope = [(s, d) for s, d in scored if d.get("segment") in (segment, "General")]
            if len(in_scope) >= top_k:
                scored = in_scope

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [doc for score, doc in scored[:top_k] if score > 0]
        return top

    def _build_query(self, segment: str, quiz_answers: Dict[str, str]) -> str:
        """Turn the segment + raw quiz answers into a natural-language
        query string for retrieval, so results reflect *how* the person
        answered, not just their final label."""
        answer_text = " ".join(quiz_answers.values())
        return f"{segment} shopper who answered: {answer_text}"

    def _compose_extractive(self, segment: str, retrieved: List[Dict]) -> Dict:
        """Fallback generation step used when no LLM is configured: cleanly
        stitches retrieved chunks into a short, cited recommendation."""
        style_note = next((d["text"] for d in retrieved if d["topic"] == "style"), None)
        strategy_note = next((d["text"] for d in retrieved if d["topic"] == "shopping_strategy"), None)
        pitfall_note = next((d["text"] for d in retrieved if d["topic"] == "pitfall"), None)
        extras = [d["text"] for d in retrieved if d["topic"] not in ("style", "shopping_strategy", "pitfall")]

        parts = []
        if style_note:
            parts.append(style_note)
        if strategy_note:
            parts.append(strategy_note)
        if pitfall_note:
            parts.append("Worth watching out for: " + pitfall_note)
        for extra in extras[:1]:
            parts.append(extra)

        return {
            "summary": " ".join(parts) if parts else "No specific guidance found for this profile yet.",
            "sources": [{"id": d["id"], "topic": d["topic"], "text": d["text"]} for d in retrieved],
        }

    def _compose_with_llm(self, segment: str, quiz_answers: Dict[str, str], retrieved: List[Dict]) -> Dict:
        """Step 3 (LLM path): ask Claude to synthesize the retrieved
        context into natural-language advice, grounded strictly in the
        provided snippets. Only used if ANTHROPIC_API_KEY is set."""
        import anthropic

        context_block = "\n".join(f"- {d['text']}" for d in retrieved)
        answers_block = "\n".join(f"- {q}: {a}" for q, a in quiz_answers.items())

        prompt = f"""You are writing a short, warm, personalized shopping-style recommendation.

The user's fashion identity segment is: {segment}

Their quiz answers were:
{answers_block}

Base your recommendation ONLY on the following retrieved reference notes.
Do not invent facts that aren't supported by them. Weave them into 3-4
natural sentences, no bullet points, no headers.

Reference notes:
{context_block}
"""

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return {
            "summary": summary.strip(),
            "sources": [{"id": d["id"], "topic": d["topic"], "text": d["text"]} for d in retrieved],
        }

    def get_recommendation(self, segment: str, quiz_answers: Dict[str, str], top_k: int = 4) -> Dict:
        """Full RAG pipeline: retrieve relevant chunks for this user, then
        generate a grounded recommendation from them."""
        query = self._build_query(segment, quiz_answers)
        retrieved = self.retrieve(query, top_k=top_k, segment=segment)

        if not retrieved:
            return {"summary": "", "sources": []}

        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                return self._compose_with_llm(segment, quiz_answers, retrieved)
            except Exception:
                # Fall back gracefully if the API call fails for any reason
                pass

        return self._compose_extractive(segment, retrieved)
