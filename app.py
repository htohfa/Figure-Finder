import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from foto import (
    MODEL_LABELS, MODEL_REGISTRY, get_model, CostTracker,
    InputParser, PaperSearcher, PaperTriager,
    PDFStore, FigureExtractor, FigureScorer,
    build_zip, format_authors, get_confidence, confidence_badge_class,
)
from foto.models import PROVIDER_DISPLAY
from foto.llm_client import LLMClient
from foto.persistence import load_stats, log_search, log_rating

st.set_page_config(
    page_title="FOTO",
    page_icon="🐇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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

.feedback-box { background: white; border: 1px solid #e8e8e0; border-radius: 4px; padding: 1.5rem 1.8rem; margin-top: 2rem; }
.feedback-title { font-family: 'DM Serif Display', serif; font-size: 1.2rem; margin-bottom: 1rem; }

.tally-box { background: #1a1a1a; color: #fafaf8; padding: 1.2rem 1.8rem; border-radius: 4px; margin-top: 1.5rem; }
.tally-title { font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; color: #888; margin-bottom: 0.8rem; }
.tally-row { display: flex; gap: 2rem; }
.tally-num { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #fafaf8; }
.tally-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #888; letter-spacing: 0.08em; margin-top: 0.2rem; }

.pathfinder-cite { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #888; line-height: 1.5; }
.pathfinder-cite a { color: #4a5568; text-decoration: underline; }

.stTextArea textarea { font-family: 'Inter', sans-serif; font-size: 0.95rem; border: 1px solid #ddd; border-radius: 3px; }
.stButton button { font-family: 'DM Mono', monospace; font-size: 0.8rem; letter-spacing: 0.06em; border-radius: 3px; }

div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label { font-family: 'DM Mono', monospace; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: #888; }

div[data-testid="stCheckbox"] label p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    color: #1a1a1a !important;
    font-weight: 400 !important;
}
</style>
""", unsafe_allow_html=True)


for key, default in {
    "pdf_cache": {},
    "results": None,
    "running": False,
    "log": [],
    "tally": {"searches": 0, "ratings": []},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
    if "global_stats" not in st.session_state:
        st.session_state.global_stats = load_stats()


st.markdown("""
<div class="foto-header">
  <p class="foto-subtitle">Figure frOm for Text & illustratiOns</p>
  <p class="foto-tagline">
    Describe a scientific figure in words, upload a sketch, or both 
    and FOTO searches the literature to find it.
  </p>
</div>
""", unsafe_allow_html=True)


col_left, col_right = st.columns([1, 1], gap="large")


def render_api_key_field(provider: str, model_cfg, key_state: str) -> str:
    display_name = PROVIDER_DISPLAY.get(provider, "API Key")
    st.markdown(f'<p class="section-label" style="margin-top:0.8rem;">{display_name}</p>', unsafe_allow_html=True)
    key = st.text_input(
        display_name, type="password", label_visibility="collapsed",
        placeholder="...", key=key_state,
    )
    st.markdown(
        f'<div class="api-help">{model_cfg.api_help_text} '
        f'<a href="{model_cfg.api_help_url}" target="_blank">Get key →</a></div>',
        unsafe_allow_html=True,
    )
    return key


with col_left:
    st.markdown('<p class="section-label">Primary Model</p>', unsafe_allow_html=True)
    primary_label = st.selectbox(
        "Primary Model", options=MODEL_LABELS,
        label_visibility="collapsed", key="primary_model",
    )
    primary_cfg = get_model(primary_label)
    primary_key = render_api_key_field(primary_cfg.provider, primary_cfg, "primary_api_key")

    use_pathfinder = st.checkbox(
        "Use Pathfinder (recommended)",
        value=True,
        key="use_pathfinder",
    )
    st.markdown(
        '<div class="pathfinder-cite" style="margin-top:-0.4rem;margin-left:1.8rem;">'
        'Based on <a href="https://arxiv.org/abs/2408.01556" target="_blank">arXiv:2408.01556</a> · '
        'OpenAI key required for query embedding · '
        '~$1 per 2M queries'
        '</div>',
        unsafe_allow_html=True,
    )

    openai_key = ""
    if use_pathfinder:
        st.markdown('<p class="section-label" style="margin-top:0.8rem;">OpenAI API Key</p>', unsafe_allow_html=True)
        openai_key = st.text_input(
            "OpenAI Key", type="password", label_visibility="collapsed",
            placeholder="sk-...", key="openai_key",
        )
        st.markdown(
            '<div class="api-help">Used to embed queries with text-embedding-3-small. '
            '<a href="https://platform.openai.com/api-keys" target="_blank">Get key →</a></div>',
            unsafe_allow_html=True,
        )

    s2_key = ""
    if not use_pathfinder:
        st.markdown('<p class="section-label" style="margin-top:0.8rem;">Semantic Scholar Key (optional)</p>', unsafe_allow_html=True)
        s2_key = st.text_input(
            "S2 Key", type="password", label_visibility="collapsed",
            placeholder="(improves keyword search)", key="s2_key",
        )
        st.markdown(
            '<div class="api-help">Optional — speeds up the keyword-based paper search. '
            '<a href="https://www.semanticscholar.org/product/api" target="_blank">Get key →</a></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-label" style="margin-top:1.2rem;">Describe the figure</p>', unsafe_allow_html=True)
    user_text = st.text_area(
        "Figure description", label_visibility="collapsed", height=120,
        placeholder='e.g. "scatter plot of cosmological parameter constraints from wavelet scattering transform, Omega_m vs sigma_8"',
    )

    st.markdown('<p class="section-label" style="margin-top:0.8rem;">Upload a sketch (optional)</p>', unsafe_allow_html=True)
    sketch_file = st.file_uploader("Sketch", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")

    run_verify = st.checkbox("Secondary verification (recommended)", value=True, key="run_verify")
    st.markdown(
        '<p style="font-size:0.78rem;color:#888;margin-top:-0.6rem;margin-left:1.8rem;">'
        'Double-checks top matches. Uses the primary model by default.</p>',
        unsafe_allow_html=True,
    )

    verify_cfg = primary_cfg
    verify_key = primary_key
    if run_verify:
        verify_options = ["Same as primary"] + MODEL_LABELS
        verify_choice = st.selectbox(
            "Verification model", options=verify_options,
            label_visibility="collapsed", key="verify_model",
        )
        if verify_choice != "Same as primary":
            verify_cfg = get_model(verify_choice)
            if verify_cfg.provider != primary_cfg.provider:
                verify_key = render_api_key_field(
                    verify_cfg.provider, verify_cfg, "verify_api_key",
                )
            else:
                verify_key = primary_key

    num_papers = st.slider("Papers to search", min_value=5, max_value=50, value=20, step=5)

    run_btn = st.button("🔭  Search", use_container_width=True, type="primary", disabled=st.session_state.running)

with col_right:
    if not st.session_state.results and not st.session_state.running:
        st.markdown("""
<div style="padding: 3rem 2rem; color: #aaa; text-align: center;">
  <div style="font-size: 3rem; margin-bottom: 1rem;">🔭</div>
  <div style="font-family: 'DM Mono', monospace; font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase;">Your search progress will appear here</div>
</div>
""", unsafe_allow_html=True)


if run_btn:
    if not primary_key:
        st.error(f"Please enter your {PROVIDER_DISPLAY[primary_cfg.provider]}.")
    elif use_pathfinder and not openai_key:
        st.error("Pathfinder is checked — please enter your OpenAI API key, or uncheck Pathfinder.")
    elif run_verify and verify_cfg.provider != primary_cfg.provider and not verify_key:
        st.error(f"Verification model needs its own key — please enter your {PROVIDER_DISPLAY[verify_cfg.provider]}.")
    elif not user_text and not sketch_file:
        st.error("Please enter a description or upload a sketch (or both).")
    else:
        st.session_state.running = True
        st.session_state.results = None

        tracker = CostTracker()
        primary_client = LLMClient(provider=primary_cfg.provider, api_key=primary_key)
        verify_client = (
            primary_client if verify_cfg.provider == primary_cfg.provider
            else LLMClient(provider=verify_cfg.provider, api_key=verify_key)
        )
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
                log("⟳ Parsing your description...")
                parser = InputParser(
                    primary_client, primary_cfg.model_id, primary_cfg.prices, tracker,
                    max_tokens=primary_cfg.triage_max_tokens * 4,
                )
                spec = parser.parse(text=user_text or None, sketch_bytes=sketch_bytes)
                query = spec["science_query"] or user_text or "(no query)"
                log(f"✓ Query: <em>{query}</em>")
                if spec.get("plot_type"):
                    log(f"  Plot type: {spec['plot_type']}")

                searcher = PaperSearcher(s2_key=s2_key or None)
                if use_pathfinder:
                    all_papers = searcher.expanded_search_pathfinder(query, openai_key, log=log)
                else:
                    all_papers = searcher.expanded_search(
                        query, primary_client, primary_cfg.model_id, primary_cfg.prices, tracker,
                        max_tokens=primary_cfg.triage_max_tokens * 4, log=log,
                    )
                log(f"✓ {len(all_papers)} unique papers found")

                log(f"⟳ Triaging papers (batches of {primary_cfg.batch_size})...")
                triager = PaperTriager(
                    primary_client, primary_cfg.model_id, primary_cfg.prices, tracker,
                    max_tokens=primary_cfg.triage_max_tokens,
                    batch_size=primary_cfg.batch_size,
                )
                triaged = triager.triage(all_papers, spec)
                top = triaged[:num_papers]
                log(f"✓ {len(top)} papers passed triage")
                paper_lookup = {p["paperId"]: p for p in top}

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

                log(f"⟳ Scoring {len(filtered)} figures (batches of {primary_cfg.batch_size})...")
                scorer = FigureScorer(
                    primary_client, primary_cfg.model_id, primary_cfg.prices, tracker,
                    score_max_tokens=primary_cfg.score_max_tokens,
                    verify_max_tokens=primary_cfg.verify_max_tokens,
                    batch_size=primary_cfg.batch_size,
                )
                results = scorer.score_batch(filtered, spec)
                primary_matches = [fig for fig, result in zip(filtered, results)
                                   if result.get("confidence", 0) >= 0.5]
                log(f"✓ {len(primary_matches)} primary matches")

                verified = primary_matches
                if run_verify and primary_matches:
                    log(f"⟳ Verifying {len(primary_matches)} matches...")
                    verifier = FigureScorer(
                        verify_client, verify_cfg.model_id, verify_cfg.prices, tracker,
                        score_max_tokens=verify_cfg.score_max_tokens,
                        verify_max_tokens=verify_cfg.verify_max_tokens,
                        batch_size=1,
                    )
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
                log_search(query, len(downloaded), len(verified), total_cost)
                st.session_state.global_stats["searches"] += 1

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
        log_rating(rating)
        st.session_state.global_stats["ratings"].append(rating)
        st.success("Thanks!")


stats = st.session_state.global_stats
n_ratings = len(stats["ratings"])
avg = sum(stats["ratings"]) / n_ratings if n_ratings else 0

st.markdown(f"""
<div class="tally-box">
  <div class="tally-title">Overall stats</div>
  <div class="tally-row">
    <div class="stat-item"><div class="tally-num">{stats['searches']}</div><div class="tally-label">Searches</div></div>
    <div class="stat-item"><div class="tally-num">{n_ratings}</div><div class="tally-label">Rated</div></div>
    <div class="stat-item"><div class="tally-num">{"—" if not n_ratings else f"{avg:.1f}"}</div><div class="tally-label">Avg score</div></div>
  </div>
</div>
""", unsafe_allow_html=True)
