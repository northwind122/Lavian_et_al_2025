from pathlib import Path


master_all = Path(__file__).resolve().parent.parent / "data"

master_motion = master_all / "visual motion"
master_landmarks = master_all / "landmarks"
master_hd = master_all / "heading direction after habenula ablation"
