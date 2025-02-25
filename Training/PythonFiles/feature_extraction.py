import os
import pandas as pd
import re
from collections import defaultdict

def extract_features(file_path, folder_name):
    """
    Extracts Fan-In, Fan-Out, and Gate Count from a Verilog file.

    Args:
    file_path (str): Path to the Verilog file
    folder_name (str): Name of the folder the file belongs to

    Returns:
    dict: Extracted feature data
    """
    try:
        with open(file_path, "r") as file:
            verilog_code = file.readlines()

        fan_in = defaultdict(int)
        fan_out = defaultdict(int)
        gate_count = 0

        # Detect gate-level instances like AND, OR, XOR, etc.
        for line in verilog_code:
            match = re.findall(r"(and|or|nand|nor|xor|xnor)\s+\w+\s*\(([\w\s,]+)\);", line)
            if match:
                for gate_type, signal_group in match:
                    signals = [sig.strip() for sig in signal_group.split(",")]
                    output_signal = signals[0]  # First signal is output
                    input_signals = signals[1:]  # Rest are inputs
                    
                    fan_in[output_signal] = len(input_signals)
                    for inp in input_signals:
                        fan_out[inp] += 1
                    gate_count += 1

        # Estimated Depth & Delay Calculation (Basic Heuristic)
        estimated_depth = min(9, max(2, gate_count // 500 + 2))  # Between 2 and 9
        estimated_delay = gate_count * 0.1  # Simple delay approximation

        return {
            "Total Fan-In": sum(fan_in.values()),
            "Total Fan-Out": sum(fan_out.values()),
            "Total Gate Count": gate_count,
            "Estimated Depth": estimated_depth,
            "Estimated Delay": round(estimated_delay, 2),
            "Filename": os.path.basename(file_path),
            "Folder": folder_name
        }

    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        return None

def process_all_folders(folder_paths, output_csv, output_excel):
    """
    Scans all Verilog files in given folders and extracts features.

    Args:
    folder_paths (list): List of folder paths to scan for Verilog files
    output_csv (str): Path to save the extracted data in CSV format
    output_excel (str): Path to save the extracted data in Excel format
    """
    extracted_data = []

    for folder in folder_paths:
        print(f"INFO: 📂 Scanning Verilog files in: {folder}")
        for root, _, files in os.walk(folder):
            for filename in files:
                if filename.endswith(".v"):  # Only Verilog files
                    file_path = os.path.join(root, filename)
                    print(f"INFO: 🔍 Processing file: {filename}")

                    features = extract_features(file_path, os.path.basename(folder))
                    if features:
                        extracted_data.append(features)
                        print(f"INFO: ✅ Extracted: {features}")

    # Convert to DataFrame
    df = pd.DataFrame(extracted_data)

    # Save as CSV
    df.to_csv(output_csv, index=False)
    print(f"✅ Data saved to CSV: {output_csv}")

    # Save as Excel (Handling Sheet Name Issue)
    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Features", index=False)
    print(f"✅ Data saved to Excel: {output_excel}")

# ✅ Main Execution
if __name__ == "__main__":
    BASE_PATH = r"D:\Anushka\Projects\Google Girl Hackathon\Benchmarks-main"
    OUTPUT_CSV = r"D:\Anushka\Projects\Google Girl Hackathon\Training\CSV Files\feature_data.csv"
    OUTPUT_EXCEL = r"D:\Anushka\Projects\Google Girl Hackathon\Training\CSV Files\feature_data.xlsx"

    # All benchmark folders
    ALL_FOLDERS = [
        os.path.join(BASE_PATH, "Combinational"),
        os.path.join(BASE_PATH, "Sequential"),
        os.path.join(BASE_PATH, "ISCAS85"),
        os.path.join(BASE_PATH, "ISCAS89")
    ]

    # Run feature extraction
    process_all_folders(ALL_FOLDERS, OUTPUT_CSV, OUTPUT_EXCEL)
