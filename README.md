# Supreme Court Legal Opinion Classification & Precedent Engine ⚖

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Course-UE24CS352A-orange.svg)](https://pes.edu/)
[![Framework](https://img.shields.io/badge/Framework-Gensim%20%7C%20PyTorch%20%7C%20Transformers-green.svg)](https://huggingface.co/)
[![Report](https://img.shields.io/badge/Report-2--Page%20PDF%20Deliverable-red.svg)](./Project_Report_Writeup.pdf)

> **Implementation, Reproduction, and Extension of:**  
> *"Classification of Legal Text"* — Krithika Iyer (ksiyer), Stanford University (CS229).  
> **Course:** UE24CS352A — Machine Learning Mini-Project  
> **Corpus:** Supreme Court Database (SCDB / Washington University via `textacy`) — 8,376 Judicial Decisions (1946–2018 Terms).

---

## 📌 Executive Summary & Problem Overview

In common law jurisprudence governed by the doctrine of **stare decisis** (precedent), prior decisions dictate the resolution of future legal disputes. Automated classification and semantic retrieval of legal opinions empower legal researchers, overburdened public defenders, and pro se litigants to search and catalog precedent efficiently.

However, judicial text presents distinct challenges:
1. **Extreme Document Length:** Opinions average ~7,000 words (up to 87,000 words) with multi-clause sentences, passive grammar, and dense citations.
2. **Subtlety of Constitutional Reasoning:** Landmark decisions often discuss facts that do not reflect their true legal grounding. For example, *Roe v. Wade* (1973) is fundamentally categorized under constitutional **Privacy** rather than health or medical law.
3. **Severe Class Imbalance:** Real-world Supreme Court dockets are heavily concentrated in Criminal Procedure, Economic Activity, and Civil Rights, while classes such as Privacy or Interstate Relations are scarce.

---

## 🚀 Key Achievements & Results

| Experiment / Track | 15 Legal Domains Acc. | 279 Issue Codes Acc. | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: |
| **LDA (50 Topics) + Logistic Regression (Baseline)** | 42.51% | 9.73% | 0.1309 | 0.3323 |
| **Doc2Vec (PV-DM) + Logistic Regression** | **70.21%** | **46.15%** | **0.6101** | **0.7027** |
| **TF-IDF (5k terms) + Logistic Regression** | 77.43% | 55.22% | 0.5410 | 0.7612 |
| **Head+Tail Contextual Transformer (BERT) + LR** | 53.60% | 22.00% | 0.4000 | 0.5092 |

### Key Findings:
- **Continuous Representations vs Topic Mixtures:** Doc2Vec (+27.7% over LDA on 15 domains) confirms that distributed paragraph vectors preserve essential legal context and semantic relationships that bag-of-words topic models discard.
- **Precedent Semantic Retrieval:** Doc2Vec embeddings accurately reconstructed the landmark *Roe v. Wade* abortion precedent network (reproducing Iyer Figure 2: *Akron*, *Thornburgh*, *Webster*, *Casey*), with *Doe v. Bolton* retrieved as the #1 nearest precedent (0.7422 cosine similarity).
- **Transformer Memory Resolution:** While Iyer (2020) encountered CUDA Out-Of-Memory failures when attempting to pass long opinions into BERT on Colab, our **Head + Tail Hierarchical Chunk Pooling** successfully extracts contextual embeddings without memory explosion.

---

## 📂 Repository Structure

```
.
├── guidelines/
│   ├── Guidelines and Instructions_ML Mini Project Assignment_2026.pdf
│   ├── Iyer.pdf                             # Original research paper
│   └── supreme-court-py3.json.gz            # SCDB corpus (8,376 decisions)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                       # Ingestion, cleaning, tokenization, label mapping
│   ├── eda.py                              # Corpus statistics, Figure 1 reproduction
│   ├── lda_model.py                         # TF-IDF & Gensim LDA topic extraction
│   ├── doc2vec_model.py                     # Gensim Doc2Vec (PV-DM) pipeline
│   ├── classifier.py                        # Logistic Regression & multi-class evaluation
│   ├── similarity_search.py                 # Precedent retrieval & Figure 2 citation network
│   ├── transformer_model.py                 # Head+Tail BERT contextual representation extractor
│   └── evaluate.py                          # Confusion matrices, Table 1, and Table 2 generators
├── outputs/
│   ├── figures/
│   │   ├── figure1_paper_distribution.png   # Reproduction of Iyer Fig. 1
│   │   ├── figure2_precedent_similarity_graph.png # Reproduction of Iyer Fig. 2
│   │   ├── confusion_matrix_doc2vec.png     # Doc2Vec 15-class heatmap
│   │   ├── confusion_matrix_lda.png         # LDA 15-class heatmap
│   │   ├── issue_area_breakdown.png         # 15 domain counts
│   │   └── corpus_length_and_timeline.png   # Length distribution & timeline
│   ├── models/                              # Trained model weights & artifacts
│   ├── table1_comparison.csv                # Overall model benchmark
│   ├── table2_breakdown.csv                 # Per-domain precision/recall/F1
│   └── metrics_summary.json                 # Complete quantitative metrics
├── web_app/
│   ├── app.py                              # Interactive Flask demonstration backend
│   ├── templates/index.html                 # Modern glassmorphic legal dashboard
│   └── static/css/style.css                 # Custom responsive styling
├── Legal_Text_Classification_End_to_End.ipynb # Interactive Jupyter Notebook
├── Project_Report_Writeup.pdf               # Mandatory 2-Page PDF submission writeup
├── Project_Report_Writeup.md                # Markdown source of the writeup
├── train_all.py                             # Master end-to-end training script
├── generate_pdf_report.py                   # PDF report compiler
├── test_app_endpoints.py                    # Automated test suite
└── requirements.txt                         # Package dependencies
```

---

## ⚡ Quick Start & Installation

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/<your-username>/supreme-court-legal-classification.git
cd supreme-court-legal-classification

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Model Pipeline
To reproduce all figures, train LDA, train Doc2Vec, evaluate classifiers, and generate comparison tables:
```bash
python train_all.py
```
*Outputs will be saved in `outputs/figures/`, `outputs/models/`, and `outputs/*.csv`.*

### 3. Launch the Interactive Web Demonstration
For the mandatory live demonstration and review panel presentation:
```bash
python web_app/app.py
```
Open your browser at: **`http://localhost:5000`**

#### Interactive Features:
1. **Corpus & EDA Tab:** Live metrics on 8,376 decisions, Figure 1 reproduction, and length distributions.
2. **Live Opinion Classifier Tab:** Select landmark cases (or enter custom legal text) and evaluate real-time classification probabilities from both Doc2Vec+LR and LDA+LR.
3. **Semantic Precedent Search Tab:** Explore the interactive Roe v. Wade citation graph (Figure 2) and run instant nearest-precedent queries.
4. **Model Benchmark Tab:** Interactive Table 1, Table 2 category breakdown, and confusion matrices.

### 4. Run the Jupyter Notebook
```bash
jupyter lab Legal_Text_Classification_End_to_End.ipynb
```

### 5. Recompile the 2-Page PDF Report
```bash
python generate_pdf_report.py
```
*Generates the compliant two-page document `Project_Report_Writeup.pdf`.*

---

## 📊 Detailed Category Breakdown (Table 2 Reproduction)

Results evaluated on the held-out test split (1,675 decisions) using **Doc2Vec + Logistic Regression**:

| Label | Legal Domain | Precision | Recall | F-1 Score | Test Support |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | Criminal Procedure | **0.84** | **0.87** | **0.85** | 385 |
| **2** | Civil Rights | **0.74** | **0.71** | **0.72** | 271 |
| **3** | First Amendment | **0.82** | **0.67** | **0.74** | 132 |
| **4** | Due Process | 0.64 | 0.37 | 0.47 | 67 |
| **5** | Privacy (*Roe v. Wade*) | 0.39 | **0.68** | **0.50** | 22 |
| **6** | Attorneys | 0.60 | **0.75** | **0.67** | 20 |
| **7** | Unions | **0.80** | 0.64 | **0.71** | 69 |
| **8** | Economic Activity | **0.72** | **0.72** | **0.72** | 333 |
| **9** | Judicial Power | 0.50 | 0.57 | 0.53 | 227 |
| **10** | Federalism | 0.49 | 0.45 | 0.47 | 73 |
| **11** | Interstate Relations | 0.47 | **0.73** | **0.57** | 11 |
| **12** | Federal Taxation | **0.77** | **0.77** | **0.77** | 61 |
| **13** | Miscellaneous | 0.17 | 0.25 | 0.20 | 4 |

---

## 🧪 Automated Testing

Execute the comprehensive test suite verifying all models, data pipelines, and API endpoints:
```bash
python test_app_endpoints.py
```

---

## 📖 Citation & References
1. **Krithika Iyer (2020)**. *Classification of Legal Text*. Stanford University (CS229 Machine Learning).
2. **Supreme Court Database (SCDB)**. Washington University School of Law. [http://scdb.wustl.edu/](http://scdb.wustl.edu/)
3. **Le, Q., & Mikolov, T. (2014)**. *Distributed Representations of Sentences and Documents*. ICML.
4. **Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003)**. *Latent Dirichlet Allocation*. JMLR.
5. **Chalkidis, I., et al. (2020)**. *LEGAL-BERT: The Muppets straight out of Law School*. EMNLP.
