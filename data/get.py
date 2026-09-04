import os
import pandas as pd

directory_path = "./"

# Columns we prefer, in order
preferred_columns = ["Correction", "Translation"]

for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        csv_path = os.path.join(directory_path, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(directory_path, txt_filename)
        
        # Read CSV with all columns
        df = pd.read_csv(csv_path)

        # Find which column to use for each row
        def pick_value(row):
            # Check preferred columns first
            for col in preferred_columns:
                if col in row and pd.notna(row[col]) and row[col] != "":
                    return row[col]
            # Otherwise fall back to *any* remaining column that has a value
            for col in row.index:
                if col not in preferred_columns and pd.notna(row[col]) and str(row[col]).strip() != "":
                    return row[col]
            return ""  # If everything is empty

        translations = df.apply(pick_value, axis=1)

        # Save to txt
        with open(txt_path, "w", encoding="utf-8") as f:
            for line in translations:
                f.write(str(line) + "\n")

        print(f"Processed: {filename}")

print("All CSV files processed.")
