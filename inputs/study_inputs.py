import pandas as pd
from pathlib import Path

def analyze_csv_files(files):
    dfs = [pd.read_csv(file) for file in files]
    df = pd.concat(dfs, ignore_index=True)

    # General Metadata
    num_rows, num_columns = df.shape
    column_info = [{"name": col, "type": str(df[col].dtype)} for col in df.columns]

    # Descriptive Statistics for String Columns
    string_columns = [col for col in df.columns if df[col].dtype == 'object']
    desc_stats_str = {}
    for col in string_columns:
        mode_val = df[col].mode()[0] if not df[col].mode().empty else "N/A"
        num_unique = df[col].nunique()
        value_counts = df[col].value_counts().head(5).to_dict()
        desc_stats_str[col] = {
            "Most Frequent Value": mode_val,
            "Number of Unique Values": num_unique,
            "Top 5 Most Frequent Values": value_counts
        }

    # Count of NULL values for each column and percentage
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / num_rows) * 100

    # Sample Data
    sample_data = df.sample(20)

    analysis = {
        "General Metadata": {
            "Number of Rows": num_rows,
            "Number of Columns": num_columns,
            "Column Information": column_info
        },
        "Descriptive Statistics for String Columns": desc_stats_str,
        "Null Value Counts": null_counts,
        "Null Value Percentages": null_percentages,
        "Sample Data": sample_data
    }

    return analysis

def write_analysis_to_txt(analysis, file_path):
    with open(file_path, "w") as f:
        f.write("=== Table Analysis ===\n\n")

        f.write("-- General Metadata --\n")
        f.write(f"Number of Rows: {analysis['General Metadata']['Number of Rows']}\n")
        f.write(f"Number of Columns: {analysis['General Metadata']['Number of Columns']}\n")
        f.write("Column Information:\n")
        for col in analysis['General Metadata']['Column Information']:
            f.write(f"{col['name']}: {col['type']}\n")
        f.write("\n")

        f.write("-- Descriptive Statistics for String Columns --\n")
        for col, stats in analysis['Descriptive Statistics for String Columns'].items():
            f.write(f"{col}:\n")
            f.write(f"  Most Frequent Value: {stats['Most Frequent Value']}\n")
            f.write(f"  Number of Unique Values: {stats['Number of Unique Values']}\n")
            f.write(f"  Top 5 Most Frequent Values: {stats['Top 5 Most Frequent Values']}\n")
        f.write("\n")

        f.write("-- Null Value Counts --\n")
        f.write(analysis['Null Value Counts'].to_string())
        f.write("\n\n")

        f.write("-- Null Value Percentages --\n")
        f.write(analysis['Null Value Percentages'].to_string())
        f.write("\n\n")

        f.write("-- Sample Data --\n")
        f.write(analysis['Sample Data'].to_string())
        f.write("\n")

if __name__ == "__main__":
    # Replace these with the paths to your CSV files
    files = [
        Path("master_beta_faltafinanciacion0.csv"),
        Path("master_beta_faltafinanciacion1.csv"),
        Path("master_beta_faltafinanciacion2.csv"),
        Path("master_beta_faltafinanciacion3.csv")
    ]

    analysis = analyze_csv_files(files)
    write_analysis_to_txt(analysis, "table_analysis.txt")
