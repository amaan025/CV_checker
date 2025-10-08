# app.py
"""
AI Resume <-> Job Description Optimizer — Production-ready single-file Streamlit app.
- Robust Hugging Face integration: prefers InferenceClient; falls back to InferenceApi with explicit task="text2text-generation".
- Retries, exponential backoff, response normalization.
- sentence-transformers semantic scoring (all-MiniLM-L6-v2).
- Safe fallbacks: if HF is not configured or fails, app returns templated rewrites so the user can continue.
IMPORTANT: Do NOT hardcode your HF token here. Set via Streamlit Secrets (TOML):
  HF_TOKEN = "hf_xxxYOURTOKENxxx"
  HF_MODEL = "google/flan-t5-small"
"""

import os
import io
import re
import time
import traceback
from typing import List, Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --------- Secrets / config (Streamlit secrets preferred) ----------
HF_TOKEN = st.secrets.get("HF_TOKEN") if "HF_TOKEN" in st.secrets else os.getenv("HF_TOKEN")
HF_MODEL = st.secrets.get("HF_MODEL") if "HF_MODEL" in st.secrets else os.getenv("HF_MODEL", "google/flan-t5-small")

# --------- Embedding model (sentence-transformers) ----------
from sentence_transformers import SentenceTransformer, util

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

embed_model = load_embedding_model()

# --------- Simple NLP utilities ----------
STOPWORDS = {
    "the","and","to","of","a","in","for","with","on","by","as","an","is","are",
    "was","were","be","that","this","it","at","from","or","into","per","via",
    "i","we","you","they","he","she","their","our","your","my"
}

SKILL_SYNONYMS = {
    "python": {"python", "py"},
    "sql": {"sql"},
    "data engineering": {"data engineering","etl","elt","airflow","pipelines"},
    "machine learning": {"machine learning","ml","sklearn","scikit-learn"},
    "deep learning": {"deep learning","pytorch","tensorflow","keras"},
    "nlp": {"nlp","natural language processing","spacy","transformers"},
    "genai": {"gen ai","genai","gpt","llm","rag","langchain"},
    "cloud": {"aws","azure","gcp","cloud"},
    "docker": {"docker","container"},
    "kubernetes": {"kubernetes","k8s"},
    "fastapi": {"fastapi","api"},
    "react": {"react","nextjs","next.js","frontend"},
    "streamlit": {"streamlit"},
    "git": {"git","github","gitlab"},
    "testing": {"pytest","unit testing","integration testing","ci","cd","cicd"},
    "financial analysis": {"finance","financial analysis","portfolio","risk","alpha"},
}

REQUIRED_HINTS = {"must", "required", "need", "essential", "mandatory"}

def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def tokenize_candidates(text: str):
    text = (text or "").lower()
    toks = re.findall(r"[a-zA-Z0-9\+\#\.]{2,}", text)
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    phrases = re.findall(r"(?:[a-zA-Z]+\s){1,4}[a-zA-Z]+", text)
    candidates = set(toks + phrases)
    return sorted(candidates)

def map_to_skills(candidates: List[str]) -> List[str]:
    normalized = set()
    cand_join = " ".join(candidates)
    for skill, variants in SKILL_SYNONYMS.items():
        if any(v in cand_join for v in variants):
            normalized.add(skill)
    for c in candidates:
        if c in {"python","sql","aws","azure","gcp","docker","kubernetes","fastapi","react","streamlit","pytorch","tensorflow","keras","langchain","faiss","airflow"}:
            normalized.add(c)
    return sorted(normalized)

def detect_required_keywords(jd_text: str) -> set:
    jd_text_low = (jd_text or "").lower()
    req = set()
    for skill, variants in SKILL_SYNONYMS.items():
        for v in variants:
            if v in jd_text_low and any(h in jd_text_low for h in REQUIRED_HINTS):
                req.add(skill)
    return req

# --------- Semantic score ----------
def semantic_cover_score(jd_text: str, cv_text: str):
    jd_chunks = [jd_text[i:i+300] for i in range(0, len(jd_text), 300)] if jd_text else []
    cv_chunks = [cv_text[i:i+600] for i in range(0, len(cv_text), 600)] if cv_text else []
    if not jd_chunks or not cv_chunks:
        return 0.0
    jd_emb = embed_model.encode(jd_chunks, convert_to_tensor=True)
    cv_emb = embed_model.encode(cv_chunks, convert_to_tensor=True)
    sims = util.cos_sim(jd_emb, cv_emb)
    best_per_jd = sims.max(dim=1).values.cpu().numpy()
    avg_sim = float(best_per_jd.mean())
    return round(avg_sim * 100, 1)

# --------- Bullet extraction ----------
def extract_bullets(cv_text: str) -> List[str]:
    lines = [l.strip() for l in cv_text.splitlines() if l.strip()]
    bullets = []
    for ln in lines:
        if ln.startswith(("-", "•", "*")) or re.match(r"^\d+[.)]\s", ln):
            bullets.append(re.sub(r"^(-|•|\*|\d+[.)])\s*", "", ln))
    if not bullets:
        parts = [p.strip() for p in re.split(r"[;\n\t]", cv_text) if len(p.strip().split()) > 6]
        bullets = parts[:8]
    return bullets[:8]

# --------- Analysis core ----------
class AnalysisResult:
    def __init__(self, jd_skills, cv_skills, missing_core, nice_to_have, score, coverage, weighted):
        self.jd_skills = jd_skills
        self.cv_skills = cv_skills
        self.missing_core = missing_core
        self.nice_to_have = nice_to_have
        self.score = score
        self.coverage = coverage
        self.weighted = weighted

def analyze_match(cv_text: str, jd_text: str) -> AnalysisResult:
    cv_text = clean_text(cv_text)
    jd_text = clean_text(jd_text)
    jd_cand = tokenize_candidates(jd_text)
    cv_cand = tokenize_candidates(cv_text)
    jd_skills = map_to_skills(jd_cand)
    cv_skills = map_to_skills(cv_cand)
    required = detect_required_keywords(jd_text)
    jd_set = set(jd_skills)
    cv_set = set(cv_skills)
    missing = list(jd_set - cv_set)
    present = list(jd_set & cv_set)
    req_present = list(required & cv_set)
    req_missing = list(required - cv_set)
    coverage = (len(present) / len(jd_set)) if len(jd_set) else 0.0
    weighted_total = len(jd_set) + len(required)
    weighted_hit = len(present) + len(req_present)
    weighted = weighted_hit / max(1, weighted_total)
    score = round(100 * (0.4 * coverage + 0.6 * weighted), 1)
    missing_core = sorted(list(req_missing))
    nice_to_have = sorted([m for m in missing if m not in missing_core])
    return AnalysisResult(
        jd_skills=jd_skills,
        cv_skills=cv_skills,
        missing_core=missing_core,
        nice_to_have=nice_to_have,
        score=score,
        coverage=round(coverage * 100, 1),
        weighted=round(weighted * 100, 1),
    )

# --------- Robust Hugging Face client + generation with fallback ----------
@st.cache_resource(show_spinner=False)
def init_hf_clients():
    """
    Initialize HF clients:
      - Prefer InferenceClient (modern)
      - Fallback to InferenceApi but pass explicit task="text2text-generation"
    Returns dict {"client": client_obj_or_None, "kind": "inferenceclient"|"inferenceapi"|None}
    """
    if not HF_TOKEN:
        return {"client": None, "kind": None}
    # Try modern client
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=HF_TOKEN)
        return {"client": client, "kind": "inferenceclient"}
    except Exception:
        # Fallback to legacy client with explicit task
        try:
            from huggingface_hub import InferenceApi
            client = InferenceApi(repo_id=HF_MODEL, token=HF_TOKEN, task="text2text-generation")
            return {"client": client, "kind": "inferenceapi"}
        except Exception:
            print("HF client init failed (both InferenceClient & InferenceApi attempted). Traceback:")
            print(traceback.format_exc())
            return {"client": None, "kind": None}

HF_CONTEXT = init_hf_clients()

def _normalize_hf_response(resp) -> str:
    try:
        if resp is None:
            return ""
        if isinstance(resp, dict):
            if "generated_text" in resp:
                return resp["generated_text"]
            for v in resp.values():
                if isinstance(v, str):
                    return v
                if isinstance(v, list) and v and isinstance(v[0], dict) and "generated_text" in v[0]:
                    return v[0]["generated_text"]
            return str(resp)
        if isinstance(resp, list):
            if resp and isinstance(resp[0], dict) and "generated_text" in resp[0]:
                return resp[0]["generated_text"]
            if resp and isinstance(resp[0], str):
                return resp[0]
            return " ".join(map(str, resp))
        if isinstance(resp, str):
            return resp
        return str(resp)
    except Exception:
        print("Normalization error:", traceback.format_exc())
        return str(resp)

def _hf_generate(prompt: str, max_new_tokens: int = 80, temperature: float = 0.2, timeout: int = 30):
    """
    Call HF generation with retries/backoff. Raises on repeated failure.
    """
    if HF_CONTEXT.get("client") is None:
        raise RuntimeError("HF_TOKEN not configured or HF client init failed.")
    client = HF_CONTEXT["client"]
    kind = HF_CONTEXT["kind"]
    attempts = 3
    backoff = 1.0
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            if kind == "inferenceclient":
                # prefer modern methods; adapt to available API
                if hasattr(client, "text_generation"):
                    resp = client.text_generation(model=HF_MODEL, inputs=prompt, max_new_tokens=max_new_tokens, temperature=temperature, timeout=timeout)
                elif hasattr(client, "generate"):
                    resp = client.generate(model=HF_MODEL, inputs=prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                else:
                    resp = client.request(model=HF_MODEL, endpoint="text-generation", inputs=prompt, parameters={"max_new_tokens": max_new_tokens, "temperature": temperature})
                return _normalize_hf_response(resp)
            elif kind == "inferenceapi":
                # legacy client
                resp = client(inputs=prompt, parameters={"max_new_tokens": max_new_tokens, "temperature": temperature})
                return _normalize_hf_response(resp)
            else:
                raise RuntimeError("No usable HF client available.")
        except Exception as e:
            last_exc = e
            print(f"HF generation attempt {attempt} failed: {repr(e)}")
            print(traceback.format_exc())
            if attempt == attempts:
                raise
            time.sleep(backoff)
            backoff *= 2.0
    raise last_exc if last_exc is not None else RuntimeError("HF generation failed")

def rewrite_bullets_with_hf(bullets: List[str], jd_text: str) -> List[str]:
    """
    Produce 1-5 rewritten bullets using HF (preferred), falling back to templated rewrites if HF is unavailable or fails.
    """
    if not bullets:
        return []
    # If HF client not configured, fallback quickly
    if HF_CONTEXT.get("client") is None:
        jd_cand = map_to_skills(tokenize_candidates(jd_text))
        out = []
        for i, b in enumerate(bullets[:5]):
            extra = (jd_cand[i % len(jd_cand)] if jd_cand else "role requirements")
            out.append(f"{b} — Aligned to {extra}; quantify impact where possible.")
        return out

    prompt_template = (
        "You are an expert resume editor. Rewrite the following resume bullet to be concise (≤24 words), "
        "start with a strong active verb, naturally include 1–2 relevant skill keywords from the job description, and emphasise measurable impact if possible.\n\n"
        "JOB DESCRIPTION:\n{jd}\n\nBULLET:\n{bullet}\n\nReturn only the rewritten bullet text (no numbering)."
    )

    rewritten = []
    for b in bullets[:5]:
        prompt = prompt_template.format(jd=jd_text, bullet=b)
        try:
            out = _hf_generate(prompt, max_new_tokens=80, temperature=0.2, timeout=30)
            out = out.replace("\n", " ").strip()
            # Keep it reasonable length
            if len(out.split()) > 40:
                out = " ".join(out.split()[:40]) + "..."
            rewritten.append(out)
        except Exception as e:
            print("HF rewrite failed for a bullet; falling back. Error:", repr(e))
            print(traceback.format_exc())
            jd_cand = map_to_skills(tokenize_candidates(jd_text))
            extra = (jd_cand[len(rewritten) % len(jd_cand)] if jd_cand else "role requirements")
            rewritten.append(f"{b} — Aligned to {extra}; quantify impact where possible.")
    return rewritten

# --------- Streamlit UI ----------
st.set_page_config(page_title="AI Resume Match", page_icon="🧠", layout="wide")

st.sidebar.title("AI Resume → JD Optimizer")
st.sidebar.markdown("Upload a CV (PDF/DOCX/TXT) and paste a job description. Get a score, gaps, and AI-tailored bullet rewrites.")
st.sidebar.markdown("---")
st.sidebar.write("HF token configured:", bool(HF_TOKEN))
st.sidebar.write("HF model:", HF_MODEL)
st.sidebar.caption("Configure HF_TOKEN & HF_MODEL in Streamlit Secrets (TOML). Do NOT hardcode secrets in code.")

st.title("AI-Powered Resume ↔ Job Description Optimizer")
st.caption("Semantic match score, missing skills, and AI-optimized bullets — robust and production-minded.")

col1, col2 = st.columns([1, 1])
with col1:
    cv_file = st.file_uploader("Upload your CV (PDF/DOCX/TXT)", type=["pdf","docx","txt"])
    cv_text_area = st.text_area("Or paste your CV text", height=220, placeholder="Paste your CV content or bullets here…")
with col2:
    jd_text_area = st.text_area("Paste the Job Description", height=420, placeholder="Paste the full job description…")

run = st.button("⚡ Analyze & Rewrite")

if run:
    cv_text = ""
    if cv_file is not None:
        # read uploaded file
        name = cv_file.name.lower()
        data = cv_file.read()
        try:
            if name.endswith(".txt"):
                cv_text = data.decode("utf-8", errors="ignore")
            elif name.endswith(".docx"):
                from docx import Document
                doc = Document(io.BytesIO(data))
                cv_text = "\n".join(p.text for p in doc.paragraphs)
            elif name.endswith(".pdf"):
                from pdfminer.high_level import extract_text
                cv_text = extract_text(io.BytesIO(data))
            else:
                cv_text = data.decode("utf-8", errors="ignore")
        except Exception:
            cv_text = data.decode("utf-8", errors="ignore")
    if not cv_text:
        cv_text = cv_text_area

    if not cv_text or not jd_text_area.strip():
        st.error("Please provide both CV text (file or paste) and a Job Description.")
        st.stop()

    # Analysis
    result = analyze_match(cv_text, jd_text_area)
    semantic_score = semantic_cover_score(jd_text_area, cv_text)

    # Score visualization
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result.score,
        delta={'reference': 70},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'thickness': 0.3},
            'steps': [
                {'range': [0, 50], 'color': "#6a2c70"},
                {'range': [50, 75], 'color': "#ffb703"},
                {'range': [75, 100], 'color': "#2a9d8f"}
            ],
            'threshold': {'line': {'color': "white", 'width': 3}, 'value': result.score}
        },
        number={'suffix': " /100"}
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))

    m1, m2, m3 = st.columns([1.2, 1, 1])
    with m1:
        st.subheader("Overall Match Score")
        st.plotly_chart(fig, use_container_width=True)
    with m2:
        st.subheader("Coverage & Weighted")
        st.metric("Coverage", f"{result.coverage}%")
        st.metric("Weighted", f"{result.weighted}%")
    with m3:
        st.subheader("Semantic Coverage")
        st.metric("Semantic Match", f"{semantic_score}%")

    st.markdown("### 🔍 Detected JD and CV Skills")
    skill_col1, skill_col2 = st.columns(2)
    with skill_col1:
        st.subheader("JD Skills")
        st.write(", ".join(result.jd_skills) if result.jd_skills else "No obvious skills detected — ensure JD text is pasted.")
    with skill_col2:
        st.subheader("CV Skills")
        st.write(", ".join(result.cv_skills) if result.cv_skills else "No skills detected in CV text. Try pasting clearly formatted bullets.")

    st.markdown("### 🔧 Skill Gaps")
    gap_c1, gap_c2 = st.columns(2)
    with gap_c1:
        st.subheader("Missing Core (Required)")
        st.write(", ".join(result.missing_core) if result.missing_core else "No required skills missing (based on heuristics).")
    with gap_c2:
        st.subheader("Nice to Have")
        st.write(", ".join(result.nice_to_have) if result.nice_to_have else "No nice-to-have skills missing.")

    # Bullets & rewrites
    st.markdown("### ✍️ AI-Tailored Bullet Rewrites")
    bullets = extract_bullets(cv_text)
    if not bullets:
        st.info("Could not detect bullets automatically. Consider pasting bullets or use '-' markers in your CV.")
    else:
        st.write("Original bullets (left) vs AI rewrites (right).")
        rewritten = rewrite_bullets_with_hf(bullets, jd_text_area)
        df = pd.DataFrame({"Original Bullet": bullets[:len(rewritten)], "AI Rewritten Bullet": rewritten})
        st.dataframe(df, use_container_width=True)
        txt = "\n".join(f"- {b}" for b in rewritten)
        st.download_button("⬇️ Download Rewritten Bullets (.txt)", txt, file_name="rewritten_bullets.txt")

    st.markdown("### 💡 Quick Action Tips")
    tips = []
    if result.missing_core:
        tips.append("Add at least one quantified bullet showing direct experience with the Missing Core skills.")
    if result.score < 70:
        tips.append("Mirror exact JD phrasing for critical skills; ATSs often match literal keywords.")
    if "testing" not in result.cv_skills:
        tips.append("Add unit/integration testing or CI/CD evidence if applicable.")
    if "cloud" in result.jd_skills and "cloud" not in result.cv_skills:
        tips.append("Mention cloud deployments and services used (AWS/Azure/GCP).")
    if not tips:
        tips = ["Tighten bullets to ≤24 words, lead with impact, and keep metrics visible."]
    for t in tips:
        st.write("- " + t)

else:
    st.info("Upload a CV and paste a Job Description, then click Analyze & Rewrite.")
