<h1 align="center">Dissertation: QPP-Based Cutoff Strategies for
Multi-Step Retrieval-Augmented
Generation</h1>

<h3 align="center">By Abdalrahman Z.M. Hameed, MSc Candidate </h3>
<h4 align="center">Acknowledgements: Debasis Ganguly, Lecturer in Data Science</h4>

---

<h2 id="project-overview">1. Project Overview</h2>
<p>
This project investigates <strong>adaptive multi-iterative Retrieval-Augmented Generation (RAG)</strong> using
query performance prediction (QPP) to decide how many reasoning iterations a system should run.
It compares <em>fixed</em> and <em>dynamic</em> cutoff strategies over multi-step question answering, evaluating how
they interact with different retrievers (BM25 and E5) in terms of both retrieval quality and answer accuracy.
</p>

<h2 id="motivation">2. Motivation</h2>
<p>
Multi-step and agentic RAG pipelines can improve reasoning on complex queries, but excessive iterations often
introduce noisy or irrelevant evidence that <em>hurts</em> performance and wastes computation.
Empirical results show that QPP scores and downstream accuracy can drop after a certain step, yet most systems
lack a principled way to detect when to stop. This project aims to better understand how reasoning quality evolves
across iterations and to design cutoff strategies that prevent over-retrieval while preserving or improving answer quality.
</p>

<h2 id="project-objectives">3. Project Objectives</h2>
<ul>
  <li>
    Implement a multi-iterative RAG baseline that combines reinforcement-learning-based reasoning
    (Search R1) with multiple retrievers (BM25 and E5).
  </li>
  <li>
    Measure iteration-wise performance using QPP scores for retrieval quality and Exact Match (EM) / F1
    for answer quality on the HotpotQA dataset.
  </li>
  <li>
    Compare <strong>fixed</strong> cutoff strategies (e.g., stop after 1–3 steps) with <strong>dynamic</strong> QPP-based cutoffs
    that terminate reasoning when QPP scores significantly decline between iterations.
  </li>
</ul>

<h2 id="architecture-overview">4. Architecture Overview</h2>
<p>
At a high level, the system implements an <strong>iterative RAG pipeline</strong> with QPP-guided cutoff analysis:
</p>
<ul>
  <li><strong>Input &amp; Retrieval:</strong> The user query is sent to a retriever (BM25 or E5) over a HotpotQA-based Wikipedia index to obtain top-k passages.</li>
  <li><strong>Multi-step Reasoning (Search R1):</strong> A reinforcement-learning-based LLM generates intermediate
      reasoning in a structured format, including <code>&lt;think&gt;</code>, <code>&lt;search&gt;</code>, and
      <code>&lt;information&gt;</code> blocks, and can iteratively reformulate queries and trigger further retrieval.</li>
  <li><strong>QPP Evaluation:</strong> At each iteration, QPP methods (NQC, A-Ratio, Spatial) estimate retrieval quality
      based on the current result list, producing a per-step quality signal.</li>
  <li><strong>Post-hoc Cutoff Analysis:</strong> A separate analysis pipeline slices the reasoning traces at different
      iteration depths (fixed cutoffs) or applies QPP-based stopping rules (dynamic cutoffs) and regenerates final
      answers using a base model (Qwen-2.5-7B-Instruct).</li>
  <li><strong>Metrics &amp; Comparison:</strong> EM and F1 scores are computed against HotpotQA gold answers, enabling a
      comparison of how different cutoff strategies and retrievers affect both accuracy and efficiency.</li>
</ul>

<h2 id="installation-usage">5. Installation &amp; Code Usage</h2>
<ol>
  <li>
    <p><strong>Clone the repository</strong></p>
    <pre><code>git clone https://github.com/abood-zm/adaptive-rag.git
cd adaptive-rag
</code></pre>
  </li>
  <li>
    <p><strong>Install dependencies</strong></p>
    <pre><code>pip install -r requirements.txt
</code></pre>
  </li>
  <li>
    <p><strong>Run post-hoc analysis (QPP-based cutoff experiments)</strong></p>
    <p>Dynamic cutoff:</p>
    <pre><code>python post-hoc-analysis.py --dynamic
</code></pre>
    <p>Static cutoff (e.g., 1-step or 3-step):</p>
    <pre><code>python post-hoc-analysis.py --cutoff 1
python post-hoc-analysis.py --cutoff 3
</code></pre>
  </li>
  <li>
    <p><strong>Run dynamic real-time experiments</strong></p>
    <pre><code>python dynamic-RT.py
</code></pre>
    <p>
      Ensure that retrieval models (BM25, E5) and the HotpotQA-based indices are correctly configured
      in your environment before running these scripts.
    </p>
  </li>
</ol>

<h2 id="file-structure">6. File Structure</h2>
<pre><code>adaptive-rag/
├── README.md
├── LICENSE
├── requirements.txt
├── get_r1_res_nq-test.py      # Baseline multi-iterative RAG pipeline (Search R1)
├── post-hoc-analysis.py       # Static & dynamic cutoff evaluation over saved traces
├── dynamic_RT.py              # Real-time QPP-guided dynamic iteration system
├── qpp_methods.py             # NQC, A-Ratio, Spatial QPP predictors
├── utils/                     # (If you add helpers, put them here)
└── data/                      # Indexes, processed datasets, traces, etc.
</code></pre>


<h2 id="license">7. License</h2>
<p>
This project is licensed under the <a href="https://github.com/abood-zm/adaptive-rag/blob/main/LICENSE">Apache License 2.0 index</a>
</p>
