"""
Exploratory Data Analysis (EDA) for the Supreme Court Legal Decisions.
Reproduces Figure 1 from Iyer (2020) and computes key corpus statistics.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.data_loader import load_supreme_court_data, ISSUE_AREA_NAMES

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'


def run_eda(df: pd.DataFrame, output_dir: str = "outputs/figures"):
    """
    Generates exploratory charts and saves them to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Corpus Statistics ---")
    print(f"Total Decisions: {len(df)}")
    print(f"Date Range: {df['decision_date'].min()} to {df['decision_date'].max()}")
    print(f"Mean Word Count: {df['word_count'].mean():.1f} words (Std: {df['word_count'].std():.1f})")
    print(f"Median Word Count: {df['word_count'].median():.1f} words")
    print(f"Max Word Count: {df['word_count'].max()} words")
    print(f"Min Word Count: {df['word_count'].min()} words")
    print(f"Total Unique Issue Areas: {df['issue_area'].nunique()}")
    print(f"Total Unique Issue Codes: {df['issue_code'].nunique()}")

    # 1. Reproduce Figure 1 from Paper: Frequency Plots of issue_code and issue_area
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=300)

    # Top: Frequency plot of issue_code
    issue_counts = df['issue_code'].value_counts().sort_index()
    x_indices = np.arange(len(issue_counts))
    ax1.plot(x_indices, issue_counts.values, color='#1f77b4', linewidth=1.2)
    ax1.fill_between(x_indices, issue_counts.values, color='#1f77b4', alpha=0.3)
    ax1.set_title("Frequency Plot: Issue Codes (Fine-Grained Legal Issues)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Issue Code Index", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_ylim(0, max(issue_counts.values) * 1.1)

    # Bottom: Frequency plot of issue_area (0 to 14)
    area_counts = df['issue_area'].value_counts().sort_index()
    all_areas = range(15)
    counts_aligned = [area_counts.get(a, 0) for a in all_areas]
    
    bars = ax2.bar(all_areas, counts_aligned, color='#2b5c8f', width=0.7, edgecolor='black', linewidth=0.8)
    ax2.set_title("Frequency Plot: Issue Area Code (15 Broad Legal Domains)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Issue Area Code", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_xticks(list(all_areas))
    ax2.set_xlim(-0.7, 14.7)

    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax2.annotate(f"{h}",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8, rotation=0)

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "figure1_paper_distribution.png")
    plt.savefig(fig1_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 1 reproduction to: {fig1_path}")

    # 2. Detailed Issue Area Breakdown with Domain Names
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    area_summary = df['issue_area_name'].value_counts()
    palette = sns.color_palette("viridis", len(area_summary))
    sns.barplot(x=area_summary.values, y=area_summary.index, palette=palette, ax=ax, edgecolor='black', linewidth=0.5)
    ax.set_title("Distribution of Legal Opinions by Curated Issue Area", fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Decisions", fontsize=12)
    ax.set_ylabel("Legal Domain", fontsize=12)
    for i, v in enumerate(area_summary.values):
        ax.text(v + 15, i, f"{v} ({v/len(df)*100:.1f}%)", va='center', fontsize=9, fontweight='semibold')
    ax.set_xlim(0, max(area_summary.values) * 1.15)
    plt.tight_layout()
    breakdown_path = os.path.join(output_dir, "issue_area_breakdown.png")
    plt.savefig(breakdown_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Issue Area breakdown to: {breakdown_path}")

    # 3. Document Length Distribution (explaining the NLP challenge / long sentence context)
    fig, (ax_w, ax_c) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    sns.histplot(df['word_count'], bins=50, kde=True, color='#2a9d8f', ax=ax_w)
    ax_w.axvline(512, color='red', linestyle='--', linewidth=1.5, label='Standard BERT Context Limit (512)')
    ax_w.axvline(df['word_count'].median(), color='navy', linestyle='-', linewidth=1.5, label=f'Median: {int(df["word_count"].median()):,} words')
    ax_w.set_title("Distribution of Word Counts per Legal Decision", fontsize=12, fontweight='bold')
    ax_w.set_xlabel("Word Count", fontsize=11)
    ax_w.set_ylabel("Frequency", fontsize=11)
    ax_w.set_xlim(0, 20000)
    ax_w.legend()

    # Timeline of Decisions
    df_with_year = df.copy()
    df_with_year['year'] = pd.to_datetime(df_with_year['decision_date'], errors='coerce').dt.year
    year_counts = df_with_year['year'].value_counts().sort_index()
    ax_c.plot(year_counts.index, year_counts.values, color='#e76f51', linewidth=1.8)
    ax_c.fill_between(year_counts.index, year_counts.values, color='#e76f51', alpha=0.25)
    ax_c.set_title("Timeline of Supreme Court Decisions (1791 - 2018)", fontsize=12, fontweight='bold')
    ax_c.set_xlabel("Year", fontsize=11)
    ax_c.set_ylabel("Number of Decisions", fontsize=11)

    plt.tight_layout()
    length_path = os.path.join(output_dir, "corpus_length_and_timeline.png")
    plt.savefig(length_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Length and Timeline plots to: {length_path}")

if __name__ == "__main__":
    df = load_supreme_court_data()
    run_eda(df)
