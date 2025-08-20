# Adaptive Multi-Iterative Retrieval-Augmented Generation (RAG)

This repository contains the code developed for the dissertation research on **Adaptive Multi-Iterative Retrieval-Augmented Generation (RAG)**. The project explores **dynamic and static cutoff strategies** for multi-step reasoning and evaluates their interaction with different retrievers (BM25 and E5) using **Query Performance Prediction (QPP)**.  

The codebase includes three main Python scripts:

---

## 1. Baseline System

**File:** `get_r1_res_nq-test.py`  

- Implements the multi-iterative RAG pipeline with the **Search R1 reinforcement learning model**.  
- Serves as the reference baseline for comparing cutoff strategies.  
- **Note:** This script was developed by a third-party PhD student and is included for reproducibility reference only.  

---

## 2. Post-hoc Analysis

**File:** `adaptive-rag.py`  

- Contains the experimental code developed for analyzing **static and dynamic cutoff strategies**.  
- Implements adaptive multi-step reasoning with multiple retrievers (BM25, E5).  
- Performs **post-hoc evaluation** on reasoning quality and answer accuracy using **QPP scores, Exact Match, and F1** metrics.  
- Supports evaluation of both **fixed** and **dynamic cutoffs**, facilitating detailed analysis of performance trade-offs.  

---

## 3. Real-Time Dynamic Cutoff

**File:** `dynamic-RT.py`  

- Extends adaptive cutoff mechanisms to **real-time question answering scenarios**.  
- Dynamically determines the number of reasoning iterations based on query complexity and retrieval quality.  
- Demonstrates the practical viability of QPP-guided adaptive reasoning in a live setting.  

---

## Research Highlights

- Investigates the **interaction between retrievers (BM25 vs E5) and multi-step reasoning**.  
- Explores the **effectiveness of QPP** in guiding adaptive iteration stopping.  
- Provides a **reproducible experimental setup** for evaluating reasoning quality, computational efficiency, and cutoff strategies.  
- Includes all scripts necessary to replicate post-hoc analyses conducted in the dissertation.  

---

## Usage

1. Clone the repository:


`git clone https://github.com/<your-username>/adaptive-rag.git`


## 2.Install dependencies:
`pip install -r requirements.txt`

## 3. Run post-hoc analysis:
### For dynamic cutoff:
`python post-hoc-analysis.py --dynamic`

### For static cutoff:
`python post-hoc-analysis.py --cutoff 1`


## 4. Run dynamic real-time experiments:
`python dynamic-RT.py`

**Note:** Ensure that retrieval models and datasets are correctly configured in the environment.


## Citation:

If you use this repository or the adaptive RAG framework in your research, please cite:

Abdalrahman, Hameed. (2025). *Adaptive Multi-Step Retrieval-Augmented Generation: A Study of Cutoff-Based Reasoning with QPP Evaluation* (Master's dissertation, University of Glasgow).
