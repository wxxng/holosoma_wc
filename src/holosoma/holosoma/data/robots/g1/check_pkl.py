import os

import joblib
import numpy as np

file_path = "/home/rllab3/Desktop/codebase/unitreeG1/holosoma/logs/obs_logs/obs_log_20260311_013901.pkl"
file_path = "/home/rllab3/Desktop/codebase/unitreeG1/holosoma_wc/logs/motion_logs/motion_log_20260311_025348.pkl"

print("CWD:", os.getcwd())
print("ABS PATH:", os.path.abspath(file_path))
print("Exists:", os.path.exists(file_path))
print("Size:", os.path.getsize(file_path), "bytes")

with open(file_path, "rb") as f:
    head = f.read(16)
    print("HEAD:", head, "HEX:", head.hex())

data = joblib.load(file_path)

print("Loaded type:", type(data))

if isinstance(data, dict):
    print("\nPer-key summary:")
    for key, value in list(data.items()):
        summary = {"type": type(value).__name__}
        try:
            if hasattr(value, "shape"):
                summary["shape"] = value.shape
            elif isinstance(value, (list, tuple)):
                summary["len"] = len(value)
            elif isinstance(value, dict):
                summary["keys"] = list(value.keys())[:10]
        except Exception as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"- {key}: {summary}")
breakpoint()    
