import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from anthropic import Anthropic

from foto import (
    MODEL_LABELS, get_model, CostTracker,
    InputParser, PaperSearcher, PaperTriager,
    PDFStore, FigureExtractor, FigureScorer,
    build_zip, format_authors, get_confidence, confidence_badge_class,
)

st.set_page_config(
    page_title="FOTO · Figure Observatory for Text & Optics",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-weight: 300; }
.stApp { background: #fafaf8; color: #1a1a1a; }

.foto-header { padding: 3rem 0 1.5rem 0; border-bottom: 1px solid #e0e0d8; margin-bottom: 2.5rem; }
.foto-title { font-family: 'DM Serif Display', serif; font-size: 2.8rem; font-weight: 400; letter-spacing: -0.02em; line-height: 1.1; color: #1a1a1a; margin: 0; }
.foto-title em { font-style: italic; color: #4a5568; }
.foto-subtitle { font-size: 0.85rem; color: #888; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 0.4rem; font-family: 'DM Mono', monospace; }
.foto-tagline { font-size: 1rem; color: #555; margin-top: 0.8rem; font-weight: 300; max-width: 560px; }

.section-label { font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; color: #999; margin-bottom: 0.5rem; }

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

.feedback-box { background: white; border: 1px solid #e8e8e0; border-radius: 4px; padding: 1.5rem 1.8rem; margin-top: 2rem; }
.feedback-title { font-family: 'DM Serif Display', serif; font-size: 1.2rem; margin-bottom: 1rem; }

.tally-box { background: #1a1a1a; color: #fafaf8; padding: 1.2rem 1.8rem; border-radius: 4px; margin-top: 1.5rem; }
.tally-title { font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; color: #888; margin-bottom: 0.8rem; }
.tally-row { display: flex; gap: 2rem; }
.tally-num { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #fafaf8; }
.tally-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #888; letter-spacing: 0.08em; margin-top: 0.2rem; }

.stTextArea textarea { font-family: 'Inter', sans-serif; font-size: 0.95rem; border: 1px solid #ddd; border-radius: 3px; }
.stButton button { font-family: 'DM Mono', monospace; font-size: 0.8rem; letter-spacing: 0.06em; border-radius: 3px; }
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stCheckbox"] label { font-family: 'DM Mono', monospace; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: #888; }
</style>
""", unsafe_allow_html=True)


# Session state
for key, default in {
    "pdf_cache": {},
    "results": None,
    "running": False,
    "log": [],
    "tally": {"searches": 0, "ratings": []},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


st.markdown("""
<div class="foto-header">
  <p class="foto-subtitle">Figure Observatory for Text &amp; Optics</p>
  <h1 class="foto-title">FOTO<em>.</em></h1>
  <p class="foto-tagline">
    Describe a scientific figure in words, upload a sketch, or both 
    and FOTO searches the literature to find it.
  </p>
</div>
""", unsafe_allow_html=True)


col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<p class="section-label">Model</p>', unsafe_allow_html=True)
    model_label = st.selectbox("Model", options=MODEL_LABELS, label_visibility="collapsed")

    st.markdown('<p class="section-label" style="margin-top:1.2rem;">Anthropic API Key</p>', unsafe_allow_html=True)
    api_key = st.text_input("Anthropic API Key", type="password", label_visibility="collapsed", placeholder="sk-ant-...")

    st.markdown('<p class="section-label" style="margin-top:0.8rem;">Semantic Scholar Key (optional)</p>', unsafe_allow_html=True)
    s2_key = st.text_input("S2 Key", type="password", label_visibility="collapsed", placeholder="(Recommended)")

    st.markdown('<p class="section-label" style="margin-top:1.5rem;">Describe the figure</p>', unsafe_allow_html=True)
    user_text = st.text_area(
        "Figure description", label_visibility="collapsed", height=120,
        placeholder='e.g. "scatter plot of cosmological parameter constraints from wavelet scattering transform, Omega_m vs sigma_8"',
    )

    st.markdown('<p class="section-label" style="margin-top:0.8rem;">Upload a sketch (optional)</p>', unsafe_allow_html=True)
    sketch_file = st.file_uploader("Sketch", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")

    run_verify = st.checkbox("Run secondary verification pass (slower, more accurate)", value=True)
    num_papers = st.slider("Papers to search", min_value=5, max_value=50, value=20, step=5)

    run_btn = st.button("🔭  Search", use_container_width=True, type="primary", disabled=st.session_state.running)

with col_right:
    if not st.session_state.results and not st.session_state.running:
        st.markdown("""
<div style="padding: 3rem 2rem; color: #aaa; text-align: center;">
  <div style="font-size: 3rem; margin-bottom: 1rem;">🔭</div>
  <div style="font-family: 'DM Mono', monospace; font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase;">Results will appear here</div>
  <div style="margin-top: 0.8rem; font-size: 0.85rem; max-width: 300px; margin-left: auto; margin-right: auto; line-height: 1.6;">
    Enter a description, an optional sketch, and your API key then hit Search (On average per search cost 20-30¢).
  </div>
</div>
""", unsafe_allow_html=True)


# Pipeline
if run_btn:
    if not api_key:
        st.error("Please enter your Anthropic API key.")
    elif not user_text and not sketch_file:
        st.error("Please enter a description or upload a sketch (or both).")
    else:
        st.session_state.running = True
        st.session_state.results = None

        model_cfg = get_model(model_label)
        client = Anthropic(api_key=api_key)
        tracker = CostTracker(model_cfg.prices)
        sketch_bytes = sketch_file.read() if sketch_file else None

        with col_right:
            log_placeholder = st.empty()
            progress_placeholder = st.empty()

            def log(msg):
                st.session_state.log.append(msg)
                log_placeholder.markdown(
                    "\n".join(f'<div class="progress-item">{m}</div>' for m in st.session_state.log[-20:]),
                    unsafe_allow_html=True,
                )

            st.session_state.log = []

            try:
                # Parse input
                log("⟳ Parsing your description...")
                parser = InputParser(client, model_cfg.smart, tracker)
                spec = parser.parse(text=user_text or None, sketch_bytes=sketch_bytes)
                query = spec["science_query"] or user_text or "(no query)"
                log(f"✓ Query: <em>{query}</em>")
                if spec.get("plot_type"):
                    log(f"  Plot type: {spec['plot_type']}")

                # Search
                log("⟳ Searching papers...")
                searcher = PaperSearcher(s2_key=s2_key or None)
                all_papers = searcher.combined_search(query, limit=num_papers)
                log(f"  {len(all_papers)} unique papers found")

                # Triage
                log("⟳ Triaging with Claude...")
                triager = PaperTriager(client, model_cfg.cheap, tracker)
                triaged = triager.triage(all_papers, spec)
                top = triaged[:num_papers]
                log(f"✓ {len(top)} papers passed triage")
                paper_lookup = {p["paperId"]: p for p in top}

                # Fetch PDFs
                log("⟳ Fetching PDFs...")
                downloaded = []
                for i, paper in enumerate(top):
                    progress_placeholder.progress((i + 1) / len(top), text=f"Fetching PDF {i+1}/{len(top)}")
                    pdf_bytes, reason = PDFStore.fetch(paper)
                    if pdf_bytes:
                        paper["_pdf_bytes"] = pdf_bytes
                        downloaded.append(paper)
                        log(f"  ✓  {paper.get('title','')[:60]}")
                    else:
                        log(f"  ✗ {paper.get('title','')[:60]}")
                    time.sleep(0.5)
                progress_placeholder.empty()
                log(f"✓ {len(downloaded)} PDFs ready")

                # Extract figures
                log("⟳ Extracting figures...")
                extractor = FigureExtractor()
                all_figures = []
                for paper in downloaded:
                    try:
                        figs = extractor.extract(paper["_pdf_bytes"], paper["paperId"])
                        all_figures.extend(figs)
                    except Exception as e:
                        log(f"  ✗ {paper.get('title','')[:40]}: {e}")

                filtered = extractor.caption_filter(all_figures, query)
                log(f"  {len(filtered)} figures after caption filter (from {len(all_figures)} total)")

                # Primary scoring
                log(f"⟳ Scoring {len(filtered)} figures...")
                scorer = FigureScorer(client, model_cfg.cheap, tracker)
                primary_matches = []
                for i, fig in enumerate(filtered):
                    progress_placeholder.progress((i + 1) / len(filtered), text=f"Scoring {i+1}/{len(filtered)}")
                    result = scorer.score(fig, spec)
                    if result.get("confidence", 0) >= 0.5:
                        primary_matches.append(fig)
                progress_placeholder.empty()
                log(f"✓ {len(primary_matches)} primary matches")

                # Verification
                verified = primary_matches
                if run_verify and primary_matches:
                    log(f"⟳ Verifying {len(primary_matches)} matches...")
                    verifier = FigureScorer(client, model_cfg.smart, tracker)
                    verified = []
                    for i, fig in enumerate(primary_matches):
                        progress_placeholder.progress((i + 1) / len(primary_matches), text=f"Verifying {i+1}/{len(primary_matches)}")
                        result = verifier.verify(fig, spec)
                        if result.get("confidence", 0) >= 0.5:
                            verified.append(fig)
                    progress_placeholder.empty()
                    log(f"✓ {len(verified)} verified matches")

                verified.sort(key=lambda m: -get_confidence(m))
                total_cost = tracker.total()
                log(f"✓ Done — ${total_cost:.4f}")

                st.session_state.results = {
                    "matches": verified,
                    "paper_lookup": paper_lookup,
                    "spec": spec,
                    "query": query,
                    "cost": total_cost,
                    "n_papers": len(downloaded),
                    "n_figures": len(all_figures),
                }

            except Exception as e:
                log(f"✗ Error: {e}")
                st.error(f"Pipeline error: {e}")
            finally:
                st.session_state.running = False

        st.rerun()


# Results
if st.session_state.results:
    res = st.session_state.results
    matches = res["matches"]
    paper_lookup = res["paper_lookup"]

    with col_right:
        st.markdown(f"""
<div class="stats-row">
  <div class="stat-item"><div class="stat-num">{len(matches)}</div><div class="stat-label">Matches</div></div>
  <div class="stat-item"><div class="stat-num">{res['n_papers']}</div><div class="stat-label">Papers</div></div>
  <div class="stat-item"><div class="stat-num">{res['n_figures']}</div><div class="stat-label">Figures</div></div>
  <div class="stat-item"><div class="stat-num">${res['cost']:.3f}</div><div class="stat-label">API Cost</div></div>
</div>
""", unsafe_allow_html=True)

        if not matches:
            st.info("No matches found. Try broadening your description or increasing the number of papers.")
        else:
            zip_bytes = build_zip(matches, paper_lookup)
            st.download_button(
                "⬇  Download all matched figures (.zip)",
                data=zip_bytes, file_name="foto_results.zip",
                mime="application/zip", use_container_width=True,
            )
            st.markdown("---")

            for i, match in enumerate(matches):
                paper = paper_lookup.get(match["paper_id"], {})
                verdict = match.get("_verify") or match.get("_primary") or {}
                conf = get_confidence(match)
                arxiv_id = paper.get("externalIds", {}).get("ArXiv")
                arxiv_link = f' · <a href="https://arxiv.org/abs/{arxiv_id}" target="_blank">arXiv:{arxiv_id}</a>' if arxiv_id else ""

                st.markdown(f"""
<div class="result-card">
  <div class="result-title">{paper.get('title', 'Unknown')[:100]}</div>
  <div class="result-meta">{format_authors(paper)} ({paper.get('year', '')}){arxiv_link}</div>
  <div class="result-badges">
    <span class="badge {confidence_badge_class(conf)}">conf {conf:.2f}</span>
    <span class="badge badge-type">{verdict.get('plot_type', '')}</span>
    <span class="badge" style="background:#f5f5f0;color:#555;">page {match['page']}</span>
  </div>
  <div style="font-size:0.82rem;color:#555;margin-bottom:0.5rem;">{verdict.get('what_is_plotted', '')}</div>
</div>
""", unsafe_allow_html=True)

                try:
                    st.image(match["image_bytes"], use_container_width=True)
                except Exception:
                    st.warning("Could not display image.")

                st.download_button(
                    "⬇ Download figure",
                    data=match["image_bytes"],
                    file_name=f"figure_{i+1:02d}_{(paper.get('title') or 'unknown')[:30].replace(' ','_')}.png",
                    mime="image/png", key=f"dl_{i}",
                )
                st.markdown("---")


# Feedback
if st.session_state.results and st.session_state.results.get("matches"):
    st.markdown("""
<div class="feedback-box">
  <div class="feedback-title">How did FOTO do?</div>
</div>
""", unsafe_allow_html=True)

    rating = st.select_slider(
        "Was one of the top matches what you were looking for? (1 = not at all, 5 = perfect match)",
        options=[1, 2, 3, 4, 5],
        value=3,
    )

    if st.button("Submit feedback", key="submit_feedback"):
        st.session_state.tally["ratings"].append(rating)
        st.session_state.tally["searches"] += 1
        st.success("Thanks!")


# Persistent tally — always visible
tally = st.session_state.tally
n_ratings = len(tally["ratings"])
avg = sum(tally["ratings"]) / n_ratings if n_ratings else 0
searches = tally.get("searches", 0)

st.markdown(f"""
<div class="tally-box">
  <div class="tally-title">Session stats</div>
  <div class="tally-row">
    <div class="stat-item"><div class="tally-num">{searches}</div><div class="tally-label">Searches</div></div>
    <div class="stat-item"><div class="tally-num">{n_ratings}</div><div class="tally-label">Rated</div></div>
    <div class="stat-item"><div class="tally-num">{"—" if not n_ratings else f"{avg:.1f}"}</div><div class="tally-label">Avg score</div></div>
  </div>
</div>
""", unsafe_allow_html=True)
