"""
Flask Web Application for Legal Text Classification & Precedent Search.
Interactive Demonstration Platform for Review and Live Evaluation.
"""

import os
import sys
import json

# Ensure project root directory is in sys.path when running script directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

from src.data_loader import ISSUE_AREA_NAMES, load_supreme_court_data, simple_tokenize
from src.doc2vec_model import LegalDoc2VecPipeline
from src.lda_model import LegalLDAPipeline
from src.classifier import LegalClassifier

app = Flask(__name__, static_folder="static", template_folder="templates")

# Global model state
doc2vec_model = None
lda_model = None
clf_d2v = None
clf_lda = None
df_cases = None
doc_id_to_case = {}


def initialize_app():
    global doc2vec_model, lda_model, clf_d2v, clf_lda, df_cases, doc_id_to_case
    print("Loading models and metadata for web application...")
    
    # Load dataset metadata (without raw text to save memory)
    df_raw = load_supreme_court_data(min_text_len=100, filter_unknown_area=True)
    df_cases = df_raw.copy()
    
    for _, row in df_cases.iterrows():
        doc_id_to_case[str(row['doc_id'])] = row.to_dict()

    models_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
    
    # Load Doc2Vec
    d2v_path = os.path.join(models_dir, "doc2vec_model.gensim")
    if os.path.exists(d2v_path):
        doc2vec_model = LegalDoc2VecPipeline.load(models_dir)
        print("Loaded Doc2Vec model.")

    # Load LDA
    lda_path = os.path.join(models_dir, "lda_model.gensim")
    if os.path.exists(lda_path):
        lda_model = LegalLDAPipeline.load(models_dir)
        print("Loaded LDA model.")

    # Load Classifiers
    clf_d2v_path = os.path.join(models_dir, "clf_d2v_area.joblib")
    if os.path.exists(clf_d2v_path):
        clf_d2v = LegalClassifier.load(clf_d2v_path)
        print("Loaded Doc2Vec Classifier.")

    clf_lda_path = os.path.join(models_dir, "clf_lda_area.joblib")
    if os.path.exists(clf_lda_path):
        clf_lda = LegalClassifier.load(clf_lda_path)
        print("Loaded LDA Classifier.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/sample_cases")
def sample_cases():
    """
    Returns curated landmark Supreme Court cases for easy testing in live demo.
    """
    landmarks = [
        {"name": "Roe v. Wade (1973)", "cite": "410 U.S. 113", "area": "Privacy", "doc_id": 3397},
        {"name": "Akron v. Akron Center for Reproductive Health (1983)", "cite": "462 U.S. 416", "area": "Privacy", "doc_id": 5094},
        {"name": "Planned Parenthood v. Casey (1992)", "cite": "505 U.S. 833", "area": "Privacy", "doc_id": 6489},
        {"name": "Halliburton Oil Well Cementing Co. v. Walker (1946)", "cite": "329 U.S. 1", "area": "Economic Activity", "doc_id": 0},
        {"name": "Monroe et al. v. Pape et al. (1961)", "cite": "365 U.S. 167", "area": "Civil Rights", "doc_id": 1610},
        {"name": "Smith v. Wade (1983)", "cite": "461 U.S. 30", "area": "Civil Rights", "doc_id": 5046}
    ]
    # Fetch text snippet for each
    results = []
    for item in landmarks:
        doc_id_str = str(item['doc_id'])
        if doc_id_str in doc_id_to_case:
            case = doc_id_to_case[doc_id_str]
            full_text = case.get('text', '')
            results.append({
                **item,
                "full_case_name": case.get('case_name'),
                "decision_date": case.get('decision_date'),
                "text_snippet": full_text[:1200] + "..." if len(full_text) > 1200 else full_text
            })
    return jsonify(results)


@app.route("/api/classify", methods=["POST"])
def classify_text():
    """
    Classifies legal opinion text using both Doc2Vec+LR and LDA+LR.
    """
    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Empty text provided"}), 400

    results = {}
    # 1. Doc2Vec classification
    if doc2vec_model and clf_d2v:
        vec = doc2vec_model.infer_vector(text).reshape(1, -1)
        pred_idx = int(clf_d2v.predict(vec)[0])
        probas = clf_d2v.predict_proba(vec)[0]
        
        # Top 3 classes
        top_indices = np.argsort(probas)[::-1][:3]
        top_classes = [
            {
                "area_id": int(clf_d2v.classes_[idx]),
                "area_name": ISSUE_AREA_NAMES.get(int(clf_d2v.classes_[idx]), f"Area {clf_d2v.classes_[idx]}"),
                "probability": round(float(probas[idx]), 4)
            }
            for idx in top_indices
        ]
        results["doc2vec"] = {
            "predicted_area_id": pred_idx,
            "predicted_area_name": ISSUE_AREA_NAMES.get(pred_idx, f"Area {pred_idx}"),
            "confidence": round(float(np.max(probas)), 4),
            "top_classes": top_classes
        }

    # 2. LDA classification
    if lda_model and clf_lda:
        lda_vec = lda_model.transform([text])
        pred_lda_idx = int(clf_lda.predict(lda_vec)[0])
        probas_lda = clf_lda.predict_proba(lda_vec)[0]
        
        results["lda"] = {
            "predicted_area_id": pred_lda_idx,
            "predicted_area_name": ISSUE_AREA_NAMES.get(pred_lda_idx, f"Area {pred_lda_idx}"),
            "confidence": round(float(np.max(probas_lda)), 4)
        }

    return jsonify(results)


@app.route("/api/precedents", methods=["GET"])
def find_precedents():
    """
    Retrieves most similar precedents for a given doc_id or citation.
    """
    doc_id = request.args.get("doc_id")
    cite = request.args.get("cite")
    topn = int(request.args.get("topn", 7))

    target_id = None
    if doc_id:
        target_id = str(doc_id)
    elif cite:
        for k, v in doc_id_to_case.items():
            if cite.lower() in (v.get('us_cite_id') or '').lower():
                target_id = k
                break

    if not target_id or not doc2vec_model:
        return jsonify({"error": "Case not found or Doc2Vec model not loaded"}), 404

    target_case = doc_id_to_case.get(target_id, {})
    similar_raw = doc2vec_model.most_similar_to_tag(target_id, topn=topn)
    
    similar_cases = []
    for tag, score in similar_raw:
        if tag in doc_id_to_case:
            c = doc_id_to_case[tag]
            similar_cases.append({
                "doc_id": int(tag),
                "similarity_score": round(float(score), 4),
                "case_name": c.get('case_name'),
                "us_cite_id": c.get('us_cite_id'),
                "decision_date": c.get('decision_date'),
                "issue_area": c.get('issue_area'),
                "issue_area_name": c.get('issue_area_name'),
                "issue_code": c.get('issue_code')
            })

    return jsonify({
        "query_case": {
            "doc_id": int(target_id),
            "case_name": target_case.get('case_name'),
            "us_cite_id": target_case.get('us_cite_id'),
            "decision_date": target_case.get('decision_date'),
            "issue_area_name": target_case.get('issue_area_name')
        },
        "similar_precedents": similar_cases
    })


@app.route("/api/metrics")
def get_metrics():
    """
    Returns metrics summary, Table 1, and Table 2.
    """
    summary_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"status": "training in progress"})


@app.route("/outputs/figures/<path:filename>")
def serve_figure(filename):
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "figures"))
    return send_from_directory(figures_dir, filename)


if __name__ == "__main__":
    initialize_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
