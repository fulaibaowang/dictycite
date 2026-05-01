# Extends Dockerfile.listwise (vLLM + rank-llm + ZephyrReranker) by adding
# the Anserini/Pyserini BM25 stack so we can run the public ragnarok-style
# baseline (BM25 -> RankZephyr listwise) end-to-end inside one image.
#
# Build (host with docker):
#   docker build --platform=linux/amd64 \
#     -f scripts/private_scripts/hpc_scripts/vega/ragnarok_baseline/Dockerfile.ragnarok_baseline \
#     -t fulaibaowang/dictycite-ragnarok:<YYYY.MM.DD> .
#
# On Vega:
#   apptainer build dictycite-ragnarok_<YYYY.MM.DD>.sif \
#     docker://fulaibaowang/dictycite-ragnarok:<YYYY.MM.DD>

ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.17.1
FROM ${VLLM_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    MPLBACKEND=Agg \
    TOKENIZERS_PARALLELISM=false \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH}

WORKDIR /app

# Java for Anserini (Pyserini wraps the Anserini fatjar via JNI).
# Pyserini 0.44+ ships Anserini 0.36 which targets Java 21; fall back to 17.
# We need the JDK (not just the JRE) because pyjnius locates the JVM by
# resolving javac — the JRE does not ship javac and pyjnius will throw
# "Unable to find javac" at import time.
#
# After install we derive JAVA_HOME from the real `java` binary and create a
# stable /usr/lib/jvm/default-java symlink: openjdk-*-jdk-headless on this
# base image does NOT set up the `default-java` convenience symlink, which
# pyjnius would otherwise expect (it looks for libjvm.so under
# $JAVA_HOME/lib/server/). Pinning a known-good symlink also keeps the path
# stable across the 21 / 17 fallback above.
RUN apt-get update && \
    (apt-get install -y --no-install-recommends openjdk-21-jdk-headless \
     || apt-get install -y --no-install-recommends openjdk-17-jdk-headless) && \
    rm -rf /var/lib/apt/lists/* && \
    JAVA_BIN="$(readlink -f "$(command -v java)")" && \
    JAVA_HOME_DIR="$(dirname "$(dirname "$JAVA_BIN")")" && \
    ln -sfn "$JAVA_HOME_DIR" /usr/lib/jvm/default-java && \
    test -f /usr/lib/jvm/default-java/lib/server/libjvm.so

ENV JAVA_HOME=/usr/lib/jvm/default-java

# Same extras as Dockerfile.listwise.
RUN uv pip install --system \
    numpy \
    pandas \
    matplotlib \
    tqdm \
    jupyterlab \
    ipykernel \
    jupytext \
    faiss-cpu

# RankLLM + its non-curated runtime deps (matches Dockerfile.listwise).
RUN uv pip install --system --no-deps rank-llm==0.25.7 && \
    python3 -m pip install \
      dacite \
      ftfy \
      openai \
      tiktoken \
      pydantic \
      requests

# Patch RankLLM identical to Dockerfile.listwise: avoid eagerly importing the
# legacy generic Reranker / LiT5 path on `from rank_llm.rerank import ...`.
RUN python3 - <<'PY'
from pathlib import Path
import shutil
import site

for p in site.getsitepackages():
    init_py = Path(p) / 'rank_llm' / 'rerank' / '__init__.py'
    if init_py.exists():
        backup = init_py.with_suffix('.py.bak')
        shutil.copy2(init_py, backup)
        text = init_py.read_text(encoding='utf-8')
        text = text.replace('from .reranker import Reranker\n', '')
        init_py.write_text(text, encoding='utf-8')
        print(f'Patched {init_py} (backup: {backup})')
        break
else:
    raise SystemExit('Could not find rank_llm/rerank/__init__.py to patch')
PY

# Anserini/Pyserini for BM25 indexing + search.
# Pyserini pulls Flask (via its server bits), which wants a newer blinker than
# the apt-installed `python3-blinker` (1.4, distutils-style). Pip refuses to
# uninstall distutils packages, so we purge it via apt first. We deliberately
# do NOT use --ignore-installed: that flag is global (not per-package) and
# would make pip re-resolve the entire dep graph, blowing away the vLLM
# image's pinned torch / transformers / nvidia-cu13 / setuptools / numpy /
# protobuf stack.
RUN apt-get update && \
    apt-get purge -y python3-blinker || true && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install "pyserini==0.44.0"

# trec_eval is handy for spot-checking; the project also has its own evaluator.
RUN python3 -m pip install pytrec_eval

# Build-time smoke test.
RUN python3 - <<'PY'
import sys
print('Python:', sys.version)

import torch
print('torch:', torch.__version__, 'cuda build:', torch.version.cuda)

import vllm
print('vllm:', vllm.__version__)

from rank_llm.data import Request, Candidate, Query  # noqa: F401
from rank_llm.rerank.listwise import ZephyrReranker  # noqa: F401
print('rank_llm.listwise.ZephyrReranker: ok')

import pyserini  # noqa: F401
from importlib.metadata import version as _pkg_version
print('pyserini:', _pkg_version('pyserini'))

from pyserini.search.lucene import LuceneSearcher  # noqa: F401
print('pyserini.LuceneSearcher: ok')

import pytrec_eval  # noqa: F401
print('pytrec_eval: ok')

print('All ragnarok-baseline imports OK')
PY

CMD ["bash", "-lc", "python3 -c 'import torch, pyserini, rank_llm; print(torch.__version__, pyserini.__version__, rank_llm.__version__)'"]
