"""FOTO webapp: caption-first figure retrieval over the astro-ph literature."""
import io
import json
import time
import zipfile
from pathlib import Path

import streamlit as st

from foto_retrieval import Retriever
from foto_figures import fetch_figures
from foto_verify import MODELS, Judge

st.set_page_config(page_title="FOTO", page_icon="🔭", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-weight: 300; }
.stApp { background: #fafaf8; color: #1a1a1a; }
.foto-header { padding: 3rem 0 1.5rem 0; border-bottom: 1px solid #e0e0d8; margin-bottom: 2.5rem; }
.foto-subtitle { font-size: 0.85rem; color: #888; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 0.4rem; font-family: 'DM Mono', monospace; }
.foto-tagline { font-size: 1rem; color: #555; margin-top: 0.8rem; font-weight: 300; max-width: 560px; }
.section-label { font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; color: #999; margin-bottom: 0.5rem; }
.api-help { font-size: 0.75rem; color: #777; margin-top: -0.5rem; margin-bottom: 0.8rem; }
.api-help a { color: #4a5568; text-decoration: underline; }
.result-card { background: white; border: 1px solid #e8e8e0; border-radius: 4px; padding: 1.2rem 1.4rem; margin-bottom: 1.2rem; }
.result-title { font-family: 'DM Serif Display', serif; font-size: 1.05rem; color: #1a1a1a; margin-bottom: 0.2rem; line-height: 1.3; }
.result-meta { font-size: 0.8rem; color: #777; font-style: italic; margin-bottom: 0.6rem; }
.result-badges { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.8rem; }
.badge { font-family: 'DM Mono', monospace; font-size: 0.68rem; padding: 0.15rem 0.5rem; border-radius: 2px; letter-spacing: 0.04em; }
.badge-high { background: #e8f4e8; color: #2d6a2d; }
.badge-mid  { background: #fef3e2; color: #8a5700; }
.badge-low  { background: #fce8e8; color: #8a2020; }
.badge-type { background: #eef2ff; color: #3d4eac; }
.stats-row { display: flex; gap: 2rem; padding: 1rem 0; border-top: 1px solid #e0e0d8; border-bottom: 1px solid #e0e0d8; margin: 1.5rem 0; }
.stat-item { text-align: center; }
.stat-num { font-family: 'DM Serif Display', serif; font-size: 1.8rem; color: #1a1a1a; line-height: 1; }
.stat-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #999; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.2rem; }
.progress-item { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #555; padding: 0.2rem 0; }
.tally-box { background: #1a1a1a; color: #fafaf8; padding: 1.2rem 1.8rem; border-radius: 4px; margin-top: 1.5rem; }
.tally-title { font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; color: #888; margin-bottom: 0.8rem; }
.tally-row { display: flex; gap: 2rem; }
.tally-num { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #fafaf8; }
.tally-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #888; letter-spacing: 0.08em; margin-top: 0.2rem; }
.stTextArea textarea { font-family: 'Inter', sans-serif; font-size: 0.95rem; border: 1px solid #ddd; border-radius: 3px; }
.stButton button { font-family: 'DM Mono', monospace; font-size: 0.8rem; letter-spacing: 0.06em; border-radius: 3px; }
div[data-testid="stSelectbox"] label, div[data-testid="stTextInput"] label { font-family: 'DM Mono', monospace; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: #888; }
</style>
""", unsafe_allow_html=True)

STATS_FILE = Path("data/stats.json")


def load_stats():
    if STATS_FILE.exists():
        return json.loads(STATS_FILE.read_text())
    return {"searches": 0, "ratings": []}


def save_stats(stats):
    STATS_FILE.parent.mkdir(exist_ok=True)
    STATS_FILE.write_text(json.dumps(stats))


for key, default in {"results": None, "running": False, "log": []}.items():
    if key not in st.session_state:
        st.session_state[key] = default
if "stats" not in st.session_state:
    st.session_state.stats = load_stats()


@st.cache_resource(show_spinner="Loading the caption index (first run only)...")
def get_retriever():
    return Retriever()


def badge_class(conf):
    return "badge-high" if conf >= 0.75 else \
           "badge-mid" if conf >= 0.5 else "badge-low"


def build_zip(matches):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, m in enumerate(matches):
            z.write(m["image_path"],
                    f"fig_{i+1:02d}_{m['arxiv_id'].replace('/', '_')}"
                    f"_f{m['fig_index']}.jpg")
    return buf.getvalue()


st.markdown("""
<div class="foto-header">
  <p class="foto-subtitle">Figure frOm Text & illustratiOns</p>
  <p class="foto-tagline">
    Describe a scientific figure in words, upload a sketch, or both —
    FOTO searches 482,750 figures from the astro-ph literature by their
    captions and verifies the matches.
  </p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<p class="section-label">Describe the figure</p>',
                unsafe_allow_html=True)
    user_text = st.text_area(
        "Figure description", label_visibility="collapsed", height=120,
        placeholder='e.g. "matter power spectrum comparison between N-body '
                    'and COLA simulations"')

    st.markdown('<p class="section-label" style="margin-top:0.8rem;">'
                'Upload a sketch (optional)</p>', unsafe_allow_html=True)
    sketch_file = st.file_uploader("Sketch", type=["png", "jpg", "jpeg", "webp"],
                                   label_visibility="collapsed")

    # verification model; DeepSeek is text-only so a sketch rules it out
    options = [m for m, cfg in MODELS.items()
               if cfg["vision"] or sketch_file is None]
    st.markdown('<p class="section-label" style="margin-top:1.2rem;">'
                'Verification model</p>', unsafe_allow_html=True)
    model_label = st.selectbox("Verification model", options=options,
                               label_visibility="collapsed")
    cfg = MODELS[model_label]
    if sketch_file is not None:
        st.markdown('<div class="api-help">Sketch uploaded — text-only '
                    'models are hidden since a sketch requires a vision '
                    'model.</div>', unsafe_allow_html=True)

    st.markdown(f'<p class="section-label" style="margin-top:0.8rem;">'
                f'{cfg["key_label"]}</p>', unsafe_allow_html=True)
    api_key = st.text_input(cfg["key_label"], type="password",
                            label_visibility="collapsed", placeholder="...")
    st.markdown(f'<div class="api-help">Retrieval is free and runs locally; '
                f'the key is only used to verify the top matches. '
                f'<a href="{cfg["key_url"]}" target="_blank">Get key →</a>'
                f'</div>', unsafe_allow_html=True)

    n_retrieve = st.slider("Figures to retrieve", 50, 200, 100, 25)
    n_verify = st.slider("Top matches to verify", 5, 50, 20, 5)

    run_btn = st.button("🔭  Search", use_container_width=True,
                        type="primary", disabled=st.session_state.running)

with col_right:
    if not st.session_state.results and not st.session_state.running:
        st.markdown("""
<div style="padding: 3rem 2rem; color: #aaa; text-align: center;">
  <div style="font-size: 3rem; margin-bottom: 1rem;">🔭</div>
  <div style="font-family: 'DM Mono', monospace; font-size: 0.75rem;
       letter-spacing: 0.1em; text-transform: uppercase;">
    Your search progress will appear here</div>
</div>""", unsafe_allow_html=True)

if run_btn:
    if not user_text.strip():
        st.error("Please describe the figure — retrieval searches captions, "
                 "so it needs at least a few words (a sketch alone is used "
                 "only at verification).")
    elif not api_key:
        st.error(f"Please enter your {cfg['key_label']} — it is only used "
                 "for verification.")
    else:
        st.session_state.running = True
        st.session_state.results = None
        sketch_bytes = sketch_file.read() if sketch_file else None

        with col_right:
            log_ph = st.empty()
            prog_ph = st.empty()
            st.session_state.log = []

            def log(msg):
                st.session_state.log.append(msg)
                log_ph.markdown(
                    "\n".join(f'<div class="progress-item">{m}</div>'
                              for m in st.session_state.log[-22:]),
                    unsafe_allow_html=True)

            try:
                t0 = time.time()
                log("⟳ Loading caption index...")
                retriever = get_retriever()
                log(f"✓ Index ready ({retriever.index.ntotal:,} figures)")

                log("⟳ Embedding query and searching...")
                cands = retriever.search(user_text.strip(), k=n_retrieve)
                log(f"✓ {len(cands)} candidates retrieved "
                    f"({time.time()-t0:.1f}s)")
                for c in cands[:3]:
                    log(f"  · {c['caption_prefix'][:70]}...")

                top = cands[:n_verify]
                log(f"⟳ Fetching {len(top)} figures from ArxivCap...")
                figs = fetch_figures(
                    [(c["arxiv_id"], c["fig_index"]) for c in top], log=log)
                have = [c for c in top
                        if (c["arxiv_id"], c["fig_index"]) in figs]
                log(f"✓ {len(have)}/{len(top)} figure images ready")

                judge = Judge(model_label, api_key)
                log(f"⟳ Verifying with {model_label}...")
                matches = []
                for i, c in enumerate(have):
                    prog_ph.progress((i + 1) / len(have),
                                     text=f"Verifying {i+1}/{len(have)}")
                    d = figs[(c["arxiv_id"], c["fig_index"])]
                    v = judge.judge(user_text.strip(), d["caption"],
                                    image_path=d["image_path"]
                                    if cfg["vision"] else None,
                                    sketch_bytes=sketch_bytes)
                    if v and v["match"] and v["confidence"] >= 0.5:
                        matches.append({**c, **d, **v})
                prog_ph.empty()
                matches.sort(key=lambda m: -m["confidence"])
                cost = judge.est_cost()
                log(f"✓ {len(matches)} verified matches — "
                    f"~${cost:.3f}, {time.time()-t0:.0f}s total")

                st.session_state.results = {
                    "matches": matches, "n_retrieved": len(cands),
                    "n_verified": len(have), "cost": cost,
                    "query": user_text.strip()}
                st.session_state.stats["searches"] += 1
                save_stats(st.session_state.stats)
            except Exception as e:
                import traceback
                traceback.print_exc()
                log(f"✗ Error: {e}")
                st.error(f"Pipeline error: {e}")
            finally:
                st.session_state.running = False
        st.rerun()

if st.session_state.results:
    res = st.session_state.results
    matches = res["matches"]
    with col_right:
        st.markdown(f"""
<div class="stats-row">
  <div class="stat-item"><div class="stat-num">{len(matches)}</div><div class="stat-label">Matches</div></div>
  <div class="stat-item"><div class="stat-num">{res['n_retrieved']}</div><div class="stat-label">Retrieved</div></div>
  <div class="stat-item"><div class="stat-num">{res['n_verified']}</div><div class="stat-label">Verified</div></div>
  <div class="stat-item"><div class="stat-num">${res['cost']:.3f}</div><div class="stat-label">API cost</div></div>
</div>""", unsafe_allow_html=True)

        if not matches:
            st.info("No verified matches. Try a broader description or "
                    "increase the retrieval depth.")
        else:
            st.download_button("⬇  Download all matched figures (.zip)",
                               data=build_zip(matches),
                               file_name="foto_results.zip",
                               mime="application/zip",
                               use_container_width=True)
            st.markdown("---")
            for i, m in enumerate(matches):
                aid = m["arxiv_id"]
                link = (f' · <a href="https://arxiv.org/abs/{aid}" '
                        f'target="_blank">arXiv:{aid}</a>')
                st.markdown(f"""
<div class="result-card">
  <div class="result-title">{(m['title'] or 'Untitled')[:110]}</div>
  <div class="result-meta">Figure {m['fig_index']}{link}</div>
  <div class="result-badges">
    <span class="badge {badge_class(m['confidence'])}">conf {m['confidence']:.2f}</span>
    <span class="badge badge-type">retrieval {m['score']:.2f}</span>
  </div>
  <div style="font-size:0.82rem;color:#555;margin-bottom:0.5rem;">{m['what_is_plotted']}</div>
</div>""", unsafe_allow_html=True)
                st.image(m["image_path"], use_container_width=True)
                with st.expander("Caption"):
                    st.write(m["caption"])
                st.download_button(
                    "⬇ Download figure",
                    data=Path(m["image_path"]).read_bytes(),
                    file_name=f"fig_{i+1:02d}_{aid.replace('/', '_')}.jpg",
                    mime="image/jpeg", key=f"dl_{i}")
                st.markdown("---")

    rating = st.select_slider(
        "Was one of the top matches what you were looking for? "
        "(1 = not at all, 5 = perfect match)",
        options=[1, 2, 3, 4, 5], value=3)
    if st.button("Submit feedback"):
        st.session_state.stats["ratings"].append(rating)
        save_stats(st.session_state.stats)
        st.success("Thanks!")

stats = st.session_state.stats
n_r = len(stats["ratings"])
avg = sum(stats["ratings"]) / n_r if n_r else 0
st.markdown(f"""
<div class="tally-box">
  <div class="tally-title">Overall stats</div>
  <div class="tally-row">
    <div class="stat-item"><div class="tally-num">{stats['searches']}</div><div class="tally-label">Searches</div></div>
    <div class="stat-item"><div class="tally-num">{n_r}</div><div class="tally-label">Rated</div></div>
    <div class="stat-item"><div class="tally-num">{'—' if not n_r else f'{avg:.1f}'}</div><div class="tally-label">Avg score</div></div>
  </div>
</div>""", unsafe_allow_html=True)
