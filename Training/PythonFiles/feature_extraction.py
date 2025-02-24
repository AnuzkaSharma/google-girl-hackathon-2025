import os
import re
import pandas as pd
import logging
from collections import defaultdict

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def extract_features_from_verilog(file_path):
    """
    Reads a Verilog file and extracts:
    - Fan-In: Number of inputs to a gate
    - Fan-Out: Number of times a signal is used as input
    - Gate Count: Total number of logic gates
    - Approximate Depth: Estimated based on number of logic levels
    - Approximate Delay: Estimated based on gate count and fan-in
    
    Args:
    file_path (str): Path to the Verilog file
    
    Returns:
    dict: Extracted Fan-In, Fan-Out, Gate Count, Depth, and Delay
    """
    try:
        with open(file_path, "r") as file:
            verilog_code = file.readlines()

        fan_in = defaultdict(int)
        fan_out = defaultdict(int)
        gate_count = 0
        depth = 0  # Estimated circuit depth
        delay = 0  # Estimated delay

        # Detect logic gates and count inputs/outputs
        for line in verilog_code:
            match = re.findall(r"(and|or|nand|nor|xor|xnor)\s+\w+\s*\(([\w\s,]+)\);", line)
            if match:
                for _, signal_group in match:
                    signals = [sig.strip() for sig in signal_group.split(",")]
                    output_signal = signals[0]  # First signal is output
                    input_signals = signals[1:]  # Remaining are inputs
                    
                    fan_in[output_signal] = len(input_signals)
                    for inp in input_signals:
                        fan_out[inp] += 1
                    gate_count += 1

                    # **Depth Approximation (Based on Fan-In)**
                    depth = max(depth, len(input_signals))

                    # **Delay Approximation (Gate Count * Fan-In Average)**
                    delay += len(input_signals) * 0.1  # Delay per gate (random assumption)

        return {
            "Total Fan-In": sum(fan_in.values()),
            "Total Fan-Out": sum(fan_out.values()),
            "Total Gate Count": gate_count,
            "Estimated Depth": depth,
            "Estimated Delay": round(delay, 3)  # Rounded delay for better readability
        }

    except FileNotFoundError:
        logging.error(f"❌ File not found: {file_path}")
        return {"Total Fan-In": 0, "Total Fan-Out": 0, "Total Gate Count": 0, "Estimated Depth": 0, "Estimated Delay": 0}
    except Exception as e:
        logging.error(f"❌ Error processing {file_path}: {str(e)}")
        return {"Total Fan-In": 0, "Total Fan-Out": 0, "Total Gate Count": 0, "Estimated Depth": 0, "Estimated Delay": 0}

def process_all_folders(folders, output_csv, output_excel):
    """
    Processes all Verilog files from multiple folders and saves extracted data to CSV and Excel.

    Args:
    folders (list): List of directories containing Verilog (.v) files
    output_csv (str): Path to save the extracted data as CSV
    output_excel (str): Path to save the formatted Excel file
    """
    data = []

    for folder in folders:
        if not os.path.exists(folder):
            logging.error(f"❌ Error: Directory {folder} does not exist.")
            continue

        logging.info(f"📂 Scanning Verilog files in: {folder}")

        # **Recursively search for `.v` files inside subdirectories**
        for root, _, files in os.walk(folder):
            for filename in files:
                if filename.endswith(".v"):  # Only process Verilog files
                    file_path = os.path.join(root, filename)
                    logging.info(f"🔍 Processing file: {filename}")

                    features = extract_features_from_verilog(file_path)
                    features["Filename"] = filename
                    features["Folder"] = os.path.basename(folder)  # Track which folder this file came from
                    data.append(features)

                    logging.info(f"✅ Extracted: {features}")

    # Save extracted data to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    logging.info(f"\n✅ Feature extraction complete! Data saved to: {output_csv}")

    # Save styled Excel file
    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Features", index=False)
        
        # Excel Formatting
        workbook = writer.book
        worksheet = writer.sheets["Features"]

        # Apply Bold Header + Auto-Width for Columns
        header_format = workbook.add_format({"bold": True, "bg_color": "#D3D3D3", "border": 1})
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(col_num, col_num, len(value) + 5)

        # Conditional Formatting (Depth + Delay)
        format_high = workbook.add_format({"bg_color": "#FF6666"})  # Red for high values
        format_low = workbook.add_format({"bg_color": "#99FF99"})  # Green for low values
        worksheet.conditional_format("D2:D100", {"type": "3_color_scale"})  # Depth Column Gradient
        worksheet.conditional_format("E2:E100", {"type": "cell", "criteria": ">", "value": 10, "format": format_high})
        worksheet.conditional_format("E2:E100", {"type": "cell", "criteria": "<", "value": 5, "format": format_low})

        # Add Chart (Line Graph for Depth)
        chart = workbook.add_chart({"type": "line"})
        chart.add_series({
            "name": "Depth",
            "categories": f"=Features!$A$2:$A${len(df)}",
            "values": f"=Features!$D$2:$D${len(df)}",
        })
        worksheet.insert_chart("H2", chart)

    logging.info(f"✅ Styled Excel saved at: {output_excel}")

# Main execution
if __name__ == "__main__":
    # ✅ List of all folders
    ALL_FOLDERS = [
        r"D:\Anushka\Projects\Google Girl Hackathon\Benchmarks-main\Combinational",
        r"D:\Anushka\Projects\Google Girl Hackathon\Benchmarks-main\Sequential",
        r"D:\Anushka\Projects\Google Girl Hackathon\Benchmarks-main\ISCAS89",
        r"D:\Anushka\Projects\Google Girl Hackathon\Benchmarks-main\ISCAS85"
    ]
    
    # ✅ Output Files
    CSV_OUTPUT_PATH = r"D:\Anushka\Projects\Google Girl Hackathon\Training\CSV Files\feature_data.csv"
    EXCEL_OUTPUT_PATH = r"D:\Anushka\Projects\Google Girl Hackathon\Training\CSV Files\feature_data.xlsx"

    process_all_folders(ALL_FOLDERS, CSV_OUTPUT_PATH, EXCEL_OUTPUT_PATH)
