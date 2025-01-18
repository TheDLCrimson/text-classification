import pandas as pd
import matplotlib.pyplot as plt

def plot_classification_results(file_path, title):
    # Load the classification results file
    results_df = pd.read_csv(file_path)

    # Extract data for plotting
    languages = results_df['Language']
    bypass_counts = results_df["1"]  # BYPASS
    reject_counts = results_df["0"]  # REJECT
    unclear_counts = results_df["-1"]  # UNCLEAR

    total_counts = bypass_counts + reject_counts + unclear_counts
    bypass_percent = (bypass_counts / total_counts * 100).round(1)
    reject_percent = (reject_counts / total_counts * 100).round(1)
    unclear_percent = (unclear_counts / total_counts * 100).round(1)

    # Create a stacked bar chart
    x = range(len(languages))
    plt.figure(figsize=(14, 7))
    plt.bar(x, bypass_counts, label="BYPASS (1)", color='green', alpha=0.7)
    plt.bar(x, reject_counts, bottom=bypass_counts, label="REJECT (0)", color='red', alpha=0.7)
    plt.bar(x, unclear_counts, bottom=[i + j for i, j in zip(bypass_counts, reject_counts)], label="UNCLEAR (-1)", color='orange', alpha=0.7)

    # Add percentage and count labels
    for i, (bypass, reject, unclear, total) in enumerate(zip(bypass_counts, reject_counts, unclear_counts, total_counts)):
        plt.text(i, bypass / 2, f"{bypass}\n({bypass_percent[i]}%)", ha='center', va='center', fontsize=9, color='white')
        plt.text(i, bypass + (reject / 2), f"{reject}\n({reject_percent[i]}%)", ha='center', va='center', fontsize=9, color='white')
        plt.text(i, bypass + reject + (unclear / 2), f"{unclear}\n({unclear_percent[i]}%)", ha='center', va='center', fontsize=9, color='white')

    # Add labels and title
    plt.xlabel("Languages", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(x, languages, rotation=45, fontsize=10)
    plt.legend(title="Classifications", fontsize=10)
    plt.tight_layout()

    # Show the graph
    plt.show()

# File paths to visualize
file_paths = [
    ("classification_flash_results_summary.csv", "Classification Results for Flash Model"),
    ("classification_flash8b_results_summary.csv", "Classification Results for Flash-8b Model")
]

# Generate plots for each file
for file_path, title in file_paths:
    plot_classification_results(file_path, title)
