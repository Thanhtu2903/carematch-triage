carematch-triage/
├─ app.py              
├─ requirements.txt        
├─ .streamlit/
│  └─ secrets.toml.example 
└─ data/
   └─ carematch_requests.csv 
data/carematch_requests.csv
# -*- coding: utf-8 -*-
# Carematch – Separate App: Generative Triage + ZIP Impact
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import faiss

# ---------------- Config ----------------
st.set_page_config(page_title="Carematch – Triage & ZIP Impact", layout="wide")
DATA_PATH = Path(__file__).resolve().parent / "data" / "carematch_requests.csv"

# ---------------- Load Data ----------------
@st.cache_data(show_spinner=True)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # light cleanup
    if "zip_code" in df.columns:
        df["zip_code"] = df["zip_code"].astype(str).str.extract(r"(\d{5})", expand=False)
    for c in ["wait_time", "urgency_score", "match_success"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

if not DATA_PATH.exists():
    st.error(f"Missing data file: {DATA_PATH}\n\nPut your CSV there and redeploy.")
    st.stop()

carematch = load_data(DATA_PATH)

# ---------------- Helpers (FAISS) ----------------
@st.cache_resource(show_spinner=True)
def build_index(texts, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return model, index

def retrieve_similar(note, model, index, df, k=5):
    q = model.encode([note], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    rows = df.iloc[I[0]].copy().reset_index(drop=True)
    rows["sim"] = D[0]
    return rows

def build_cases_summary(similar_rows: pd.DataFrame, max_len: int = 220) -> str:
    lines = []
    for _, r in similar_rows.iterrows():
        summ = str(r.get("condition_summary", ""))[:max_len].replace("\n"," ")
        lines.append(
            f"- summary: {summ} | urgency: {r.get('urgency_score','')} | "
            f"specialty: {r.get('provider_specialty','')} | wait_time: {r.get('wait_time','')} | "
            f"match_success: {r.get('match_success','')} | sim={r.get('sim',0):.3f}"
        )
    return "\n".join(lines)

# ---------------- LLM (optional) ----------------
PROMPT_TMPL = """
You are a clinical triage assistant that reads a patient intake note and recommends:
1) triage category (e.g., 'Routine', 'Urgent', 'Immediate / ED referral'),
2) recommended specialty to route to,
3) recommended next step (telehealth, in-office, ED, call nurse),
4) brief rationale and any safety caveats.

Patient note:
{note}

Similar historical cases (most similar first):
{cases_summary}

Respond in strict JSON with fields:
- triage (string: 'Routine' | 'Urgent' | 'Immediate / ED referral')
- specialty (string)
- next_step (string)
- rationale (string)
- confidence_reason (string)
If uncertain or risk detected, choose the safer option and include 'ESCALATE' in rationale.
"""

def call_openai_json(prompt: str) -> dict | None:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=300, temperature=0.1
        )
        content = resp.choices[0].message.content
        try:
            return json.loads(content)
        except Exception:
            s, e = content.find("{"), content.rfind("}")
            if s >= 0 and e > s:
                return json.loads(content[s:e+1])
            return None
    except Exception as e:
        st.warning(f"OpenAI call failed: {e}")
        return None

# Safety rules (quick)
RED_FLAGS = [
    r"chest pain",
    r"shortness of breath|dyspnea",
    r"stroke|facial droop|weakness (?:on )?one side|slurred speech|aphasia",
    r"uncontrolled bleeding|vomiting blood|black tarry stool",
    r"severe headache|worst headache",
    r"major trauma|fracture|loss of consciousness",
]

import re
def rule_urgency(text: str) -> str:
    t = (text or "").lower()
    for p in RED_FLAGS:
        if re.search(p, t):
            return "Immediate / ED referral"
    return "Routine"

DEFAULT_MAPPING = [
    (r"acne|psoriasis|eczema", "Dermatology"),
    (r"shortness of breath|dyspnea|asthma", "Pulmonology"),
    (r"chest pain|angina|palpitation", "Cardiology"),
    (r"anxiety|depression|panic", "Behavioral Health"),
    (r"diabetes|hyperglycemia", "Endocrinology"),
    (r"pregnan|obstetric|gynecolog", "Obstetrics & Gynecology"),
    (r"back pain|sciatica", "Orthopedics"),
    (r"uti|dysuria|urinary tract", "Primary Care"),
]
def map_specialty(text: str) -> str:
    t = (text or "").lower()
    for pat, spec in DEFAULT_MAPPING:
        if re.search(pat, t, flags=re.IGNORECASE):
            return spec
    return "Primary Care"

def safe_merge(offline_triage: str, offline_spec: str, llm_obj: dict | None) -> dict:
    out = {
        "triage": offline_triage,
        "specialty": offline_spec,
        "next_step": "in-office visit",
        "rationale": "Rule-based baseline",
        "confidence_reason": "Conservative default"
    }
    if not llm_obj:
        return out
    priority = {"Routine":1, "Urgent":2, "Immediate / ED referral":3}
    triage_llm = llm_obj.get("triage", offline_triage)
    if priority.get(triage_llm,0) < priority.get(offline_triage,0):
        out["rationale"] = f"Kept conservative rule-based triage over LLM: {triage_llm} < {offline_triage}"
        return out
    out.update({
        "triage": triage_llm,
        "specialty": llm_obj.get("specialty", out["specialty"]),
        "next_step": llm_obj.get("next_step", out["next_step"]),
        "rationale": llm_obj.get("rationale", out["rationale"]),
        "confidence_reason": llm_obj.get("confidence_reason", out["confidence_reason"]),
    })
    return out

# ---------------- UI ----------------
st.title("🧠 Carematch – Triage & ZIP Impact")

tab1, tab2 = st.tabs(["Triage AI (RAG)", "ZIP Impact Analytics"])

# ===== TAB 1: TRIAGE =====
with tab1:
    st.subheader("Patient Intake")
    colA, colB = st.columns([1.2, 1])
    with colA:
        note = st.text_area("Patient note / Condition summary", height=160,
            placeholder="e.g., Sharp chest pain on deep breaths, 45yo, HTN history.")
        use_llm = st.toggle("Use Generative LLM (needs OPENAI_API_KEY in Secrets)", value=False)
        k = st.slider("Top-K similar cases", 3, 15, 5)
        model_name = st.selectbox("Embedding model", ["all-MiniLM-L6-v2"], index=0)
        run = st.button("Generate triage")

    with colB:
        st.info("Outputs will show here once you click **Generate triage**.")

    st.divider()

    if run and note.strip():
        with st.spinner("Building index & retrieving similar cases..."):
            model, index = build_index(carematch["condition_summary"].fillna("").astype(str).tolist(), model_name)
            similar = retrieve_similar(note, model, index, carematch, k=k)
            st.subheader("Most similar historical cases")
            st.dataframe(similar[["condition_summary","provider_specialty","urgency_score","wait_time","match_success","sim"]],
                        use_container_width=True, height=260)

        offline_triage = rule_urgency(note)
        offline_spec = map_specialty(note)
        prompt = PROMPT_TMPL.format(note=note, cases_summary=build_cases_summary(similar))
        llm_obj = call_openai_json(prompt) if use_llm else None
        result = safe_merge(offline_triage, offline_spec, llm_obj)

        st.subheader("Triage result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Triage", result["triage"])
        c2.metric("Specialty", result["specialty"])
        c3.metric("Next step", result["next_step"])
        st.write("**Rationale:**", result["rationale"])
        st.caption(f"Confidence: {result['confidence_reason']}")
        with st.expander("Prompt (RAG)"):
            st.code(prompt, language="markdown")
        with st.expander("Raw JSON"):
            st.json(result)

# ===== TAB 2: ZIP IMPACT =====
with tab2:
    st.subheader("How ZIP code impacts access & performance")

    if "zip_code" not in carematch.columns:
        st.warning("No `zip_code` column in the dataset.")
    else:
        df = carematch.copy()
        # build core KPIs per ZIP
        g = df.groupby("zip_code", dropna=True)
        kpis = g.agg(
            n_cases=("condition_summary","count"),
            wait_mean=("wait_time","mean"),
            wait_median=("wait_time","median"),
            wait_p90=("wait_time", lambda s: np.nanpercentile(pd.to_numeric(s, errors="coerce").dropna(), 90)),
            success_rate=("match_success","mean"),
            urgency_avg=("urgency_score","mean")
        ).reset_index()

        # ZIP Impact Index (0-100): higher = worse access (long waits & low success)
        # normalize wait_mean ↑ and (1 - success_rate) ↑ then combine
        for col in ["wait_mean","success_rate"]:
            if col not in kpis.columns:
                kpis[col] = np.nan

        # handle missing safely
        kpis["wait_norm"] = (kpis["wait_mean"] - kpis["wait_mean"].min()) / (kpis["wait_mean"].max() - kpis["wait_mean"].min() + 1e-9)
        kpis["fail_norm"] = (1 - kpis["success_rate"]).clip(lower=0, upper=1)
        kpis["impact_index"] = (0.7 * kpis["wait_norm"] + 0.3 * kpis["fail_norm"]) * 100

        st.markdown("**ZIP-level KPIs**")
        st.dataframe(
            kpis.sort_values(["impact_index","wait_mean"], ascending=[False, False])[["zip_code","n_cases","wait_mean","wait_median","wait_p90","success_rate","impact_index"]],
            use_container_width=True, height=350
        )

        # Filters / Focus
        cols = st.columns(3)
        with cols[0]:
            top_n = st.slider("Show top N worst ZIPs by Impact Index", 5, 50, 10)
        top = kpis.sort_values("impact_index", ascending=False).head(top_n)

        # Bar charts
        fig1, ax1 = plt.subplots(figsize=(9,4))
        ax1.bar(top["zip_code"], top["impact_index"])
        ax1.set_title("Impact Index (higher = worse)")
        ax1.set_ylabel("Index (0–100)")
        ax1.set_xlabel("ZIP")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(9,4))
        ax2.bar(top["zip_code"], top["wait_mean"])
        ax2.set_title("Average wait time by ZIP")
        ax2.set_ylabel("Days (mean)")
        ax2.set_xlabel("ZIP")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig2)

        # Specialty mix per ZIP (optional small table)
        if "provider_specialty" in df.columns:
            st.markdown("**Specialty concentration (top specialties per ZIP)**")
            mix = (df
                   .groupby(["zip_code","provider_specialty"])
                   .size()
                   .reset_index(name="count"))
            mix["share"] = mix.groupby("zip_code")["count"].transform(lambda s: s/s.sum())
            topmix = (mix.sort_values(["zip_code","share"], ascending=[True, False])
                        .groupby("zip_code")
                        .head(3))
            st.dataframe(topmix, use_container_width=True, height=260)

        st.caption(
            "Impact Index = 70% normalized mean wait + 30% failure (1 - success rate). "
            "Use it to triage operational focus: high-index ZIPs likely need more capacity or faster routing."
        )
