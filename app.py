# app.py
"""
AI Resume ↔ Job Description Optimizer — production-ready single-file Streamlit app.

Features:
- Robust Hugging Face integration (InferenceClient preferred; InferenceApi fallback).
- Flexible HF call surface using inspect.signature to avoid argument mismatch errors.
- Retries/backoff, timeouts, and graceful heuristic fallback.
- Improved bullet extraction (skip contact/header lines, merge broken lines).
- Generic few-shot prompt examples (domain-agnostic) to improve rewrite quality for any JD/CV.
- Caching for embedding model.
- No secrets hard-coded. Use Streamlit Secrets (TOML) or environment variables.
"""

import os
import io
import re
import time
import inspect
import traceback
from typing import List

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ----------------------------
# Config: secrets & defaults
# ----------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN") if "HF_TOKEN" in st.secrets else os.getenv("HF_TOKEN")
HF_MODEL = st.secrets.get("HF_MODEL") if "HF_MODEL" in st.secrets else os.getenv("HF_MODEL", "google/flan-t5-small")

# ----------------------------
# Embeddings (sentence-transformers)
# ----------------------------
from sentence_transformers import SentenceTransformer, util

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

embed_model = load_embedding_model()

# ----------------------------
# Simple NLP helpers
# ----------------------------
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

# ----------------------------
# Semantic coverage score
# ----------------------------
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

# ----------------------------
# Improved bullet extraction
# ----------------------------
_CONTACT_SIGNS = ["@", "linkedin", "github", "github:", "tel:", "phone", "+44", "+1", "email", "cv", "resume", "address"]

def extract_bullets_improved(cv_text: str, keep_min_words: int = 4) -> List[str]:
    if not cv_text:
        return []
    lines = cv_text.splitlines()
    cleaned = [re.sub(r"\s+", " ", ln).strip() for ln in lines if ln.strip()]
    candidates = []
    for ln in cleaned:
        low = ln.lower()
        # Skip obvious contact lines
        if any(sig in low for sig in _CONTACT_SIGNS):
            continue
        # Explicit bullets / numbered lines
        if ln.startswith(("-", "•", "*")) or re.match(r"^\d+[.)]\s", ln):
            text = re.sub(r"^(-|•|\*|\d+[.)])\s*", "", ln).strip()
            candidates.append(text)
        else:
            candidates.append(ln)

    # Merge continuation lines: if a line doesn't end punctuation and next line doesn't begin like a new bullet, merge
    merged = []
    i = 0
    while i < len(candidates):
        cur = candidates[i].strip()
        # skip very short non-bullet fragments
        if len(cur.split()) < 2:
            i += 1
            continue
        j = i + 1
        while j < len(candidates):
            nxt = candidates[j].strip()
            if not nxt:
                j += 1
                continue
            # If current ends with ., !, ? consider it complete
            if re.search(r"[\.!\?]$", cur):
                break
            # If next looks like it starts a new bullet (capital + verb or numbered), stop merging
            if re.match(r"^\d+[.)]\s", nxt) or re.match(r"^[A-Z][a-z]+", nxt) and len(nxt.split()) > 3 and nxt[0].isupper():
                break
            # Else merge
            cur = cur + " " + nxt
            j += 1
        merged.append(cur.strip())
        i = j

    # Filter and clean
    final = []
    for m in merged:
        m2 = re.sub(r"^[\-\*\u2022\d\)\.]+", "", m).strip()
        m2 = re.sub(r"\s+", " ", m2)
        m2 = re.sub(r"[\—\–\-\.\,]{1,}$", "", m2).strip()
        if len(m2.split()) >= keep_min_words:
            final.append(m2)
    # dedupe preserving order
    seen = set()
    out = []
    for f in final:
        k = f.lower()
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out[:8]

# ----------------------------
# Analysis core
# ----------------------------
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

# ----------------------------
# Robust Hugging Face client + flexible caller
# ----------------------------
@st.cache_resource(show_spinner=False)
def init_hf_client_context():
    """
    Initialize HF client context. Returns {"client": obj|None, "kind": "inferenceclient"|"inferenceapi"|None}
    """
    if not HF_TOKEN:
        return {"client": None, "kind": None}
    # try modern InferenceClient
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=HF_TOKEN)
        return {"client": client, "kind": "inferenceclient"}
    except Exception:
        # fallback to InferenceApi with explicit task
        try:
            from huggingface_hub import InferenceApi
            client = InferenceApi(repo_id=HF_MODEL, token=HF_TOKEN, task="text2text-generation")
            return {"client": client, "kind": "inferenceapi"}
        except Exception:
            print("HF client init failed. Traceback:")
            print(traceback.format_exc())
            return {"client": None, "kind": None}

HF_CONTEXT = init_hf_client_context()

def _normalize_hf_response(resp) -> str:
    try:
        if resp is None:
            return ""
        if isinstance(resp, dict):
            if "generated_text" in resp:
                return resp["generated_text"]
            # handle common nested forms
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

def _call_hf_flexible(client, kind, prompt, max_new_tokens, temperature, timeout):
    """
    Inspect available methods on client and call one that matches signature.
    This avoids TypeError("unexpected keyword 'inputs'") across hf versions.
    """
    callables = []
    if kind == "inferenceclient":
        for name in ("text_generation", "text2text", "generate", "request", "text_generation_stream"):
            if hasattr(client, name):
                callables.append(getattr(client, name))
    elif kind == "inferenceapi":
        # InferenceApi is itself callable: client(inputs=..., parameters=...)
        callables.append(client)

    # include generic request if available and not in list
    if hasattr(client, "request") and getattr(client, "request") not in callables:
        callables.append(getattr(client, "request"))

    last_exc = None
    for fn in callables:
        try:
            sig = inspect.signature(fn)
            params = sig.parameters
            kwargs = {}
            # model param
            if "model" in params:
                kwargs["model"] = HF_MODEL
            # prompt name variants
            if "inputs" in params:
                kwargs["inputs"] = prompt
            elif "input" in params:
                kwargs["input"] = prompt
            elif "prompt" in params:
                kwargs["prompt"] = prompt
            # token/length params
            if "max_new_tokens" in params:
                kwargs["max_new_tokens"] = max_new_tokens
            if "max_tokens" in params and "max_new_tokens" not in params:
                kwargs["max_tokens"] = max_new_tokens
            if "temperature" in params:
                kwargs["temperature"] = float(temperature)
            if "timeout" in params:
                kwargs["timeout"] = int(timeout)
            # Try calling with kwargs
            try:
                return fn(**kwargs)
            except TypeError:
                # try positional fallback
                pos_args = []
                for n in ("model","inputs","input","prompt","max_new_tokens","max_tokens","temperature"):
                    if n in params:
                        if n in ("inputs","input","prompt"):
                            pos_args.append(prompt)
                        elif n in ("max_new_tokens","max_tokens"):
                            pos_args.append(max_new_tokens)
                        elif n == "temperature":
                            pos_args.append(temperature)
                        elif n == "model":
                            pos_args.append(HF_MODEL)
                return fn(*pos_args)
        except Exception as e:
            last_exc = e
            print(f"Candidate callable {fn} failed with: {repr(e)}")
            print(traceback.format_exc())
            continue
    raise last_exc if last_exc is not None else RuntimeError("No suitable HF callable found")

def _hf_generate(prompt: str, max_new_tokens: int = 80, temperature: float = 0.2, timeout: int = 30):
    if HF_CONTEXT.get("client") is None:
        raise RuntimeError("HF_TOKEN not configured or HF client init failed.")
    client = HF_CONTEXT["client"]
    kind = HF_CONTEXT["kind"]
    attempts = 3
    backoff = 1.0
    last_exc = None
    for attempt in range(1, attempts+1):
        try:
            resp = _call_hf_flexible(client, kind, prompt, max_new_tokens, temperature, timeout)
            return _normalize_hf_response(resp)
        except Exception as e:
            last_exc = e
            print(f"HF generation attempt {attempt} failed: {repr(e)}")
            print(traceback.format_exc())
            if attempt == attempts:
                raise
            time.sleep(backoff)
            backoff *= 2.0
    raise last_exc if last_exc is not None else RuntimeError("HF generation failed")

# ----------------------------
# Generic few-shot examples to improve generic rewrite quality (not tailored to any single JD)
# ----------------------------
GENERIC_EXAMPLES = [
    # (short JD hint, example bullet, example rewrite)
    ("Backend development, Python, Docker",
     "Wrote background tasks and processed job queues using celery to handle tasks",
     "Built Celery workers to process background jobs reliably, improving throughput and reducing backlog."),
    ("Deployment, CI/CD, AWS",
     "Set up deployment pipelines in GitHub Actions and deployed microservices",
     "Implemented CI/CD pipelines with GitHub Actions to deploy microservices, enabling repeatable releases."),
    ("Data processing, ETL, SQL",
     "Cleaned and transformed datasets to be used by analytics team",
     "Developed ETL pipelines to clean and transform data for analytics, improving data quality."),
]

PROMPT_SUFFIX = (
    "Rewrite rules: (1) <=24 words, (2) start with a strong action verb, (3) include 1-2 relevant JD keywords if present, "
    "(4) show measurable impact when available, (5) output single-line bullet with no numbering."
)

def compose_prompt_generic(jd_text: str, bullet: str) -> str:
    jd_clean = clean_text(jd_text)
    b_clean = clean_text(bullet)
    parts = []
    for hint, ex_b, ex_r in GENERIC_EXAMPLES:
        parts.append(f"JD example (hint): {hint}\nBullet: {ex_b}\nRewrite: {ex_r}\n")
    parts.append(f"JOB DESCRIPTION:\n{jd_clean}\n")
    parts.append(f"ORIGINAL BULLET:\n{b_clean}\n")
    parts.append(PROMPT_SUFFIX)
    return "\n".join(parts)

# ----------------------------
# Heuristic fallback rewrite (produces professional single-line bullets)
# ----------------------------
_TECH_KEYWORDS = set([
    "python","java","javascript","node","c#","csharp","react","sql","aws","azure","gcp",
    "docker","kubernetes","airflow","git","github","ci","cd","spark","pandas","numpy",
    "scikit-learn","pytorch","tensorflow","fastapi","flask"
])

_VERB_PRIORITY = ["improved","reduced","built","implemented","developed","designed","deployed","led","automated","optimised","optimized","created","streamlined","engineered","spearheaded","maintained"]

def heuristic_rewrite(bullet: str, jd_text: str) -> str:
    b = re.sub(r"\s+", " ", bullet).strip()
    low = b.lower()

    # metric extraction
    metric = None
    pct = re.search(r"(\d{1,3}%|\d+\.\d+%|\breduced by \d{1,3}%|\bincreased by \d{1,3}%|\b\d{1,3}\s?%|\b\d{1,3}\s?per cent\b)", low)
    if pct:
        metric = pct.group(0)
    else:
        num = re.search(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?k?m?|\b\d{2,6}\b(?:\s?(?:rows|users|requests|transactions|events|customers))?)", low)
        if num:
            metric = num.group(0)

    # tech detection
    found_techs = [t for t in _TECH_KEYWORDS if t in low]
    tech_str = ", ".join(found_techs[:2]) if found_techs else ""

    # verb selection
    verb = None
    for v in _VERB_PRIORITY:
        if v in low:
            verb = v
            break
    if not verb:
        tokens = re.findall(r"[a-zA-Z]+", low)
        for tok in tokens[:6]:
            if tok not in ("the","a","an","and","with","for","to","in","on","by","of","using"):
                verb = tok
                break
    if not verb:
        verb = "Contributed to"

    # object extraction
    obj = None
    m = re.search(r"\b" + re.escape(verb) + r"\b(.*)", low)
    if m:
        tail = m.group(1).strip()
        if tail:
            obj = " ".join(tail.split()[:8])
    if not obj:
        obj = " ".join(low.split()[:6])

    pieces = []
    pieces.append(verb.capitalize())
    if obj:
        pieces.append(obj)
    if tech_str:
        pieces.append("using " + tech_str)
    if metric:
        pieces.append("— " + metric)
    else:
        if any(w in low for w in ("reduc","improv","optimi","save","cut","lower")):
            pieces.append("— improved efficiency")
        elif any(w in low for w in ("deploy","release","deliver","ship")):
            pieces.append("— improved delivery")
    out = " ".join(pieces)
    out = re.sub(r"\s+", " ", out).strip()
    if not out.endswith("."):
        out = out + "."
    words = out.split()
    if len(words) > 28:
        out = " ".join(words[:28]) + "..."
    return out.capitalize()

# ----------------------------
# Main robust rewrite function (HF + heuristic fallback)
# ----------------------------
def rewrite_bullets_robust(original_text: str, jd_text: str) -> List[str]:
    bullets = extract_bullets_improved(original_text)
    if not bullets:
        return []

    outputs = []
    for b in bullets[:8]:
        if len(b.split()) < 3:
            continue
        sanitized_b = re.sub(r"\S+@\S+|\bhttps?://\S+|www\.\S+", "", b).strip()
        # build prompt with generic few-shot examples
        prompt = compose_prompt_generic(jd_text, sanitized_b)
        if HF_CONTEXT.get("client") is None:
            outputs.append(heuristic_rewrite(b, jd_text))
            continue
        try:
            gen = _hf_generate(prompt, max_new_tokens=64, temperature=0.12, timeout=20)
            if not gen or len(gen.strip()) < 6:
                raise RuntimeError("Empty/invalid HF response")
            gen = gen.replace("\n", " ").strip()
            if len(gen.split()) > 32:
                gen = " ".join(gen.split()[:32]) + "..."
            outputs.append(gen)
        except Exception as e:
            print("HF failed for bullet; falling back:", repr(e))
            print(traceback.format_exc())
            outputs.append(heuristic_rewrite(b, jd_text))
    return outputs

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Resume Match", page_icon="🧠", layout="wide")

st.sidebar.title("AI Resume → JD Optimizer")
st.sidebar.markdown("Upload a CV (PDF/DOCX/TXT) or paste text, add a job description, then get a match score and AI rewrites.")
st.sidebar.markdown("---")
st.sidebar.write("HF token configured:", bool(HF_TOKEN))
st.sidebar.write("HF model:", HF_MODEL)
st.sidebar.caption("Set HF_TOKEN and HF_MODEL in Streamlit Secrets (TOML). Do not hardcode tokens in code.")

st.title("AI Resume ↔ Job Description Optimizer")
st.caption("Semantic match score, skill gaps, and professional bullet rewrites (generic — works across domains).")

col1, col2 = st.columns([1, 1])
with col1:
    cv_file = st.file_uploader("Upload your CV (PDF/DOCX/TXT)", type=["pdf","docx","txt"])
    cv_text_area = st.text_area("Or paste your CV text", height=220, placeholder="Paste your CV or bullets here…")
with col2:
    jd_text_area = st.text_area("Paste the Job Description", height=420, placeholder="Paste a full job description here…")

run = st.button("⚡ Analyze & Rewrite")

if run:
    cv_text = ""
    if cv_file is not None:
        data = cv_file.read()
        name = cv_file.name.lower()
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

    # Visual score indicator
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
    sc1, sc2 = st.columns(2)
    with sc1:
        st.subheader("JD Skills")
        st.write(", ".join(result.jd_skills) if result.jd_skills else "No obvious skills detected.")
    with sc2:
        st.subheader("CV Skills")
        st.write(", ".join(result.cv_skills) if result.cv_skills else "No skills detected in CV text.")

    st.markdown("### 🔧 Skill Gaps")
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Missing Core (Required)")
        st.write(", ".join(result.missing_core) if result.missing_core else "No required skills missing.")
    with g2:
        st.subheader("Nice to Have")
        st.write(", ".join(result.nice_to_have) if result.nice_to_have else "No nice-to-have skills missing.")

    # Bullets and rewrites
    st.markdown("### ✍️ AI-Tailored Bullet Rewrites")
    rewritten = rewrite_bullets_robust(cv_text, jd_text_area)
    if not rewritten:
        st.info("Could not extract bullets automatically. Try adding '-' markers or paste bullets explicitly.")
    else:
        df = pd.DataFrame({"AI Rewritten Bullet": rewritten})
        st.dataframe(df, width='stretch')
        download_txt = "\n".join(f"- {b}" for b in rewritten)
        st.download_button("⬇️ Download Rewritten Bullets (.txt)", download_txt, file_name="rewritten_bullets.txt")

    st.markdown("### 💡 Quick Action Tips")
    tips = []
    if result.missing_core:
        tips.append("Add at least one quantified bullet showing direct experience with missing core skills.")
    if result.score < 70:
        tips.append("Mirror exact JD phrasing for critical skills; many ATS match literal keywords.")
    if "testing" not in result.cv_skills:
        tips.append("Add unit/integration testing or CI/CD evidence if relevant.")
    if "cloud" in result.jd_skills and "cloud" not in result.cv_skills:
        tips.append("Mention cloud deployments and services used (AWS/Azure/GCP).")
    if not tips:
        tips = ["Tighten bullets to ≤24 words, lead with impact, and keep metrics visible."]
    for t in tips:
        st.write("- " + t)

else:
    st.info("Upload a CV and paste a Job Description, then click Analyze & Rewrite.")
