"""
Precedent Similarity Search and Graph Visualization.
Reproduces Figure 2 from Iyer (2020) using Doc2Vec embeddings.
Enables semantic search for relevant case law precedents.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from src.doc2vec_model import LegalDoc2VecPipeline


# Landmark abortion decisions examined in Figure 2
FIGURE_2_CASES = [
    {
        "short_name": "Roe v. Wade (1973)",
        "full_name": "ROE et al. v. WADE, DISTRICT ATTORNEY OF DALLAS COUNTY",
        "citation": "410 U.S. 113",
        "year": 1973,
        "paper_score": 1.000,
        "pos": (0.8, -0.6)
    },
    {
        "short_name": "Akron v. Akron Center (1983)",
        "full_name": "CITY OF AKRON v. AKRON CENTER FOR REPRODUCTIVE HEALTH, INC., et al.",
        "citation": "462 U.S. 416",
        "year": 1983,
        "paper_score": 0.645,
        "pos": (0.4, 0.7)
    },
    {
        "short_name": "Thornburgh v. ACOG (1986)",
        "full_name": "THORNBURGH, GOVERNOR OF PENNSYLVANIA, et al. v. AMERICAN COLLEGE OF OBSTETRICIANS AND GYNECOLOGISTS et al.",
        "citation": "476 U.S. 747",
        "year": 1986,
        "paper_score": 0.617,
        "pos": (0.05, -0.8)
    },
    {
        "short_name": "Webster v. RHS (1989)",
        "full_name": "WEBSTER, ATTORNEY GENERAL OF MISSOURI, et al. v. REPRODUCTIVE HEALTH SERVICES et al.",
        "citation": "492 U.S. 490",
        "year": 1989,
        "paper_score": 0.567,
        "pos": (-0.6, 0.6)
    },
    {
        "short_name": "Planned Parenthood v. Casey (1992)",
        "full_name": "PLANNED PARENTHOOD OF SOUTHEASTERN PENNSYLVANIA, et al. v. ROBERT P. CASEY, et al., ETC.",
        "citation": "505 U.S. 833",
        "year": 1992,
        "paper_score": 0.512,
        "pos": (-0.75, -0.4)
    }
]


class LegalSimilarityEngine:
    def __init__(self, doc2vec_pipeline: LegalDoc2VecPipeline, metadata_df: pd.DataFrame):
        self.doc2vec = doc2vec_pipeline
        self.df = metadata_df
        # Create quick lookups
        self.tag_to_row = {}
        for row_idx, row in self.df.iterrows():
            doc_id = str(row['doc_id'])
            self.tag_to_row[doc_id] = row

    def find_case_by_citation(self, citation: str) -> Optional[pd.Series]:
        """
        Locates a case record given a citation string (e.g., '410 U.S. 113').
        """
        matches = self.df[self.df['us_cite_id'].fillna('').str.contains(citation, case=False, regex=False)]
        if len(matches) > 0:
            return matches.iloc[0]
        return None

    def search_similar_precedents(
        self,
        doc_id: int,
        topn: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Finds the top N most similar legal cases to a given doc_id.
        """
        raw_similar = self.doc2vec.most_similar_to_tag(doc_id, topn=topn)
        results = []
        for tag, score in raw_similar:
            if tag in self.tag_to_row:
                row = self.tag_to_row[tag]
                results.append({
                    "doc_id": int(tag),
                    "similarity_score": round(float(score), 4),
                    "case_name": row['case_name'],
                    "us_cite_id": row['us_cite_id'],
                    "decision_date": row['decision_date'],
                    "issue_area": row['issue_area'],
                    "issue_area_name": row['issue_area_name'],
                    "issue_code": row['issue_code']
                })
        return results

    def compute_figure2_subgraph(self, output_path: str = "outputs/figures/figure2_precedent_similarity_graph.png"):
        """
        Extracts the 5 abortion landmark cases from the dataset, calculates their pairwise
        Doc2Vec cosine similarity, and renders a network diagram reproducing Figure 2.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Locate each case in the dataframe
        case_records = []
        for c in FIGURE_2_CASES:
            row = self.find_case_by_citation(c['citation'])
            if row is not None:
                case_records.append({**c, "doc_id": str(row['doc_id'])})
            else:
                print(f"Warning: Could not find case with citation {c['citation']}")

        if len(case_records) < 2:
            print("Cannot generate Figure 2 graph: insufficient cases found.")
            return

        # Compute pairwise cosine similarities
        G = nx.Graph()
        for c in case_records:
            G.add_node(
                c['short_name'],
                citation=c['citation'],
                year=c['year'],
                doc_id=c['doc_id'],
                paper_score=c['paper_score'],
                pos=c['pos']
            )

        roe_case = case_records[0] # Roe v. Wade
        roe_id = roe_case['doc_id']

        # Add edges from Roe to each other decision, plus interconnected edges
        for other in case_records[1:]:
            sim = self.doc2vec.similarity_between_tags(roe_id, other['doc_id'])
            G.add_edge(
                roe_case['short_name'],
                other['short_name'],
                weight=sim,
                paper_weight=other['paper_score']
            )

        # Also add cross edges
        for i in range(1, len(case_records)):
            for j in range(i + 1, len(case_records)):
                sim = self.doc2vec.similarity_between_tags(
                    case_records[i]['doc_id'],
                    case_records[j]['doc_id']
                )
                G.add_edge(case_records[i]['short_name'], case_records[j]['short_name'], weight=sim)

        # Plot network graph
        fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
        pos = {n: G.nodes[n]['pos'] for n in G.nodes()}

        # Draw edges
        for u, v, data in G.edges(data=True):
            is_roe_edge = (u == roe_case['short_name'] or v == roe_case['short_name'])
            color = '#1f77b4' if is_roe_edge else '#b0bec5'
            width = 2.5 if is_roe_edge else 1.0
            style = 'solid' if is_roe_edge else 'dotted'
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], ax=ax,
                edge_color=color, width=width, style=style, alpha=0.85
            )
            # Label Roe edges with computed similarity score and original paper score
            if is_roe_edge:
                mid_x = (pos[u][0] + pos[v][0]) / 2.0
                mid_y = (pos[u][1] + pos[v][1]) / 2.0
                sim_val = data['weight']
                paper_val = data.get('paper_weight', None)
                label_txt = f"{sim_val:.3f}"
                if paper_val:
                    label_txt += f"\n(Paper: {paper_val:.3f})"
                ax.text(
                    mid_x, mid_y, label_txt,
                    fontsize=8.5, fontweight='bold', color='#004080',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#e6f2ff", edgecolor="#1f77b4", alpha=0.9),
                    ha='center', va='center'
                )

        # Draw nodes
        node_colors = ['#ff7f0e' if n == roe_case['short_name'] else '#2ca02c' for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=1200, edgecolors='black', linewidths=1.5)

        # Draw labels
        labels = {}
        for n, d in G.nodes(data=True):
            labels[n] = f"{n}\n[{d['citation']}]"
        
        # Position labels slightly offset from nodes
        label_pos = {n: (p[0], p[1] + 0.12 if p[1] >= 0 else p[1] - 0.14) for n, p in pos.items()}
        for node, text in labels.items():
            ax.text(
                label_pos[node][0], label_pos[node][1], text,
                fontsize=9.5, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#888888", alpha=0.9)
            )

        ax.set_title(
            "Semantic Precedent Network: Abortion Decisions Similar to Roe v. Wade (410 U.S. 113)\n"
            "Cosine Similarity based on Doc2Vec Latent Embeddings (Reproducing Iyer Fig. 2)",
            fontsize=12, fontweight='bold', pad=15
        )
        ax.set_xlim(-1.15, 1.25)
        ax.set_ylim(-1.15, 1.05)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved Figure 2 precedent graph to {output_path}")
