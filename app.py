# app.py
import os
import io
import re
import time
import json
from typing import List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dotenv import load_dotenv
load_dotenv()  # loads .env in local dev if present

# -----------------------------
# Config / Tokens
# -----------------------------
# Read token from environment (local) or Streamlit secrets
HF_TOKEN = os.getenv("HF_TOKEN") or (st.secrets.get("HF_TOKEN") if "HF_TOKEN" in st.secrets else None)
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-large")
USE_HF = bool(HF_TOKEN)

# -----------------------------
# Optional: local model for embeddings
# -----------------------------
from sentence_transformers import SentenceTransformer, util
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

embed_model = load_embedding_model()

# -----------------------------
# Hugging Face Inference Client (lazy)
# -----------------------------
_hf_client = None
def get_hf_client(model_id=HF_MODEL):
    global _hf_client
    if _hf_client is None:
        from huggingface_hub import InferenceApi
        _hf_client = InferenceApi(repo_id=model_id, token=HF_TOKEN)
    return _hf_client

# -----------------------------
# Simple NLP helpers (no heavy spaCy dependency)
# -----------------------------
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

def extract_text_from_upload(file) -> str:
    name = file.name.lower()
    data = file.read()
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".docx"):
        from docx import Document
        bio = io.BytesIO(data)
        doc = Document(bio)
        return "\n".join(p.text for p in doc.paragraphs)
    if name.endswith(".pdf"):
        from pdfminer.high_level import extract_text
        bio = io.BytesIO(data)
        return extract_text(bio)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def tokenize_candidates(text: str):
    text = (text or "").lower()
    toks = re.findall(r"[a-zA-Z0-9\+\#\.]{2,}", text)
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    # also include multi-word phrases (naive)
    phrases = re.findall(r"(?:[a-zA-Z]+\s){1,4}[a-zA-Z]+", text)
    candidates = set(toks + phrases)
    return sorted(candidates)

def map_to_skills(candidates: List[str]) -> List[str]:
    normalized = set()
    cand_set = set(candidates)
    for skill, variants in SKILL_SYNONYMS.items():
        if any(v in " ".join(cand_set) for v in variants):
            normalized.add(skill)
    for c in cand_set:
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

# -----------------------------
# Semantic coverage score (sentence-transformers)
# -----------------------------
def semantic_cover_score(jd_text: str, cv_text: str):
    # chunk inputs if too long
    jd_chunks = [jd_text[i:i+300] for i in range(0, len(jd_text), 300)] if jd_text else []
    cv_chunks = [cv_text[i:i+600] for i in range(0, len(cv_text), 600)] if cv_text else []

    if not jd_chunks or not cv_chunks:
        return 0.0

    jd_emb = embed_model.encode(jd_chunks, convert_to_tensor=True)
    cv_emb = embed_model.encode(cv_chunks, convert_to_tensor=True)

    sims = util.cos_sim(jd_emb, cv_emb)  # shape (len(jd), len(cv))
    best_per_jd = sims.max(dim=1).values.cpu().numpy()
    avg_sim = float(best_per_jd.mean())  # 0..1
    return round(avg_sim * 100, 1)

# -----------------------------
# Bullet extraction
# -----------------------------
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

# -----------------------------
# HF-based rewrite (with fallback)
# -----------------------------
def rewrite_bullets_with_hf(bullets: List[str], jd_text: str, model_id=HF_MODEL):
    if not bullets:
        return []
    if not USE_HF:
        # simple fallback
        jd_cand = map_to_skills(tokenize_candidates(jd_text))
        out = []
        for i, b in enumerate(bullets[:5]):
            extra = (jd_cand[i % len(jd_cand)] if jd_cand else "role requirements")
            out.append(f"{b} (Aligned to {extra}; quantify impact where possible.)")
        return out

    client = get_hf_client(model_id=model_id)
    prompt_template = (
        "You are an expert resume editor. Rewrite the following resume bullet to be concise (<=24 words), "
        "start with a strong active verb, naturally include relevant skill keywords from the job description, and emphasise measurable impact where possible.\n\n"
        "JOB DESCRIPTION:\n"
        "{jd}\n\nBULLET:\n{bullet}\n\nReturn the rewritten bullet only."
    )

    rewritten = []
    for b in bullets[:5]:
        prompt = prompt_template.format(jd=jd_text, bullet=b)
        try:
            resp = client(inputs=prompt, parameters={"max_new_tokens": 80, "temperature": 0.2})
            # resp may be a dict with 'generated_text' or a string
            if isinstance(resp, dict) and "generated_text" in resp:
                text = resp["generated_text"]
            elif isinstance(resp, list) and len(resp) and isinstance(resp[0], dict) and "generated_text" in resp[0]:
                text = resp[0]["generated_text"]
            elif isinstance(resp, str):
                text = resp
            else:
                text = str(resp)
            text = text.strip().replace("\n", " ")
            rewritten.append(text)
            time.sleep(0.15)
        except Exception as e:
            st.warning(f"Hugging Face call error (falling back): {e}")
            rewritten.append(f"{b} (tailored to JD; improve with HF token).")
    return rewritten

# -----------------------------
# Analysis core: map skills + compute coverage/score
# -----------------------------
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

    # weight required skills more
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

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Resume Match", page_icon="🧠", layout="wide")

# Sidebar
st.sidebar.title("AI Resume → JD Optimizer")
st.sidebar.markdown("Upload your CV (PDF/DOCX/TXT) and paste a job description. Get a score, gaps, and AI-tailored bullet rewrites.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model settings**")
st.sidebar.write(f"Hugging Face token present: **{bool(USE_HF)}**")
if USE_HF:
    st.sidebar.write(f"HF model: **{HF_MODEL}** (change by setting HF_MODEL env var)")
st.sidebar.caption("Note: Don’t commit tokens. Use platform secrets for deployment.")

st.title("AI-Powered Resume ↔ Job Description Optimizer")
st.caption("Semantic match score, missing skills, and AI-optimized bullets — demo-ready.")

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
        cv_text = extract_text_from_upload(cv_file)
    if not cv_text:
        cv_text = cv_text_area

    if not cv_text or not jd_text_area.strip():
        st.error("Please provide both CV text (file or paste) and a Job Description.")
        st.stop()

    # analysis
    result = analyze_match(cv_text, jd_text_area)
    semantic_score = semantic_cover_score(jd_text_area, cv_text)

    # Visual score ring
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
        if result.jd_skills:
            st.write(", ".join(result.jd_skills))
        else:
            st.write("No obvious skills detected — ensure JD text is pasted.")
    with skill_col2:
        st.subheader("CV Skills")
        if result.cv_skills:
            st.write(", ".join(result.cv_skills))
        else:
            st.write("No skills detected in CV text. Try pasting clearly formatted bullets.")

    st.markdown("### 🔧 Skill Gaps")
    gap_c1, gap_c2 = st.columns(2)
    with gap_c1:
        st.subheader("Missing Core (Required)")
        if result.missing_core:
            st.write(", ".join(result.missing_core))
        else:
            st.write("No required skills missing (based on heuristics).")
    with gap_c2:
        st.subheader("Nice to Have")
        if result.nice_to_have:
            st.write(", ".join(result.nice_to_have))
        else:
            st.write("No nice-to-have skills missing.")

    # Bullets and rewrites
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
