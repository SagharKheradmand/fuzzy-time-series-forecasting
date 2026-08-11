from __future__ import annotations
from typing import List
import ast

from fts_model import FLRGModel


def run_cli(model: FLRGModel) -> None:
    print("\nFTS Prediction Interface")
    print(f"Model order = {model.order}")
    print("Enter historical values as a Python-like list, e.g. [120, 125, 128]")
    print("Type 'exit' to quit.\n")

    while True:
        s = input("history> ").strip()
        if s.lower() in {"exit", "quit"}:
            break
        try:
            vals = ast.literal_eval(s)
            if not isinstance(vals, list) or len(vals) < model.order:
                print(f"Please enter a list with at least {model.order} numbers.")
                continue
            vals = [float(x) for x in vals]
            pred = model.predict_next_from_history(vals)
            print(f"predicted_next = {pred:.6f}\n")
        except Exception as e:
            print(f"Error: {e}\n")
