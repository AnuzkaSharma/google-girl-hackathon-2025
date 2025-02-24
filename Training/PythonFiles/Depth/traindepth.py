import os
import re
from collections import defaultdict

def analyze_verilog(file_path):
    with open(file_path, "r") as file:
        verilog_code = file.readlines()

    fan_in = defaultdict(int)
    fan_out = defaultdict(int)
    gate_count = 0
    wire_signals = set()

    # Extract wire signals
    for line in verilog_code:
        wire_match = re.search(r"wire\s+([\w\s,]+);", line)
        if wire_match:
            wires = wire_match.group(1).replace(" ", "").split(",")
            wire_signals.update(wires)

    # Detect gate-level instances like "and a1(C, a, b);" or "xor x1(S, a, b);"
    for line in verilog_code:
        match = re.findall(r"(and|or|nand|nor|xor|xnor)\s+\w+\s*\(([\w\s,]+)\);", line)
        if match:
            for gate_type, signal_group in match:
                signals = [sig.strip() for sig in signal_group.split(",")]
                output_signal = signals[0]  # First signal is the output
                input_signals = signals[1:]  # Rest are inputs
                
                fan_in[output_signal] = len(input_signals)  # Inputs ka count
                for inp in input_signals:
                    fan_out[inp] += 1  # Inputs ka fan-out badhao
                gate_count += 1

    return fan_in, fan_out, gate_count

# ✅ Teri exact directory set kar di
verilog_folder = r"D:\Anushka\Projects\Google Girl Hackathon\Benchmarks-main\Combinational"

# ✅ Folder ke andar saari .v files scan karne ka loop
for filename in os.listdir(verilog_folder):
    if filename.endswith(".v"):  # Sirf Verilog (.v) files check karega
        file_path = os.path.join(verilog_folder, filename)
        print(f"\n📂 Processing file: {filename}")

        # ✅ Function ko call kar aur output print kar
        fan_in, fan_out, gate_count = analyze_verilog(file_path)

        print("📌 Fan-In:", dict(fan_in))
        print("📌 Fan-Out:", dict(fan_out))
        print("📌 Total Gate Count:", gate_count)
