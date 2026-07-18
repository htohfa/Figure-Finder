# FOTO

FOTO Huggingface Space deployment
Caption-first figure retrieval over 482,750 astro-ph figures (paper-matching pipeline: bge query embedding -> FAISS index -> ArxivCap figure fetch -> LLM verification).
Files: app.py, foto_retrieval.py, foto_figures.py, foto_verify.py, requirements.txt.
Setup on a HF Space (Streamlit SDK)


Cold start downloads the index (~1.5 GB from htohfa/foto-index) and thebge model (~0.4 GB); allow a few minutes on first boot. CPU basic is fine.\
Users paste their own verification key (Anthropic / OpenAI / DeepSeek); retrieval itself is local and free.
