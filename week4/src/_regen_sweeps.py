"""One-off: regenerate only fig_rq3_sweeps.pdf/png without recomputing every arm."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import RunConfig, load_labelled
from run_all import SWEEP_ARMS
from analysis import Evaluator, load_arm, fig_sweeps

config = RunConfig()
frame = load_labelled(config)
evaluator = Evaluator(frame)
row_ids = frame.index.to_numpy()

arms = ["mlp_dro", *SWEEP_ARMS]
nested = {}
for arm in arms:
    loaded = load_arm(config, "nested", arm, row_ids)
    if loaded is None:
        print(f"MISSING {arm}")
        continue
    nested[arm] = evaluator.nested(loaded)
    print(f"done {arm}", flush=True)

fig_sweeps(nested)
print("wrote fig_rq3_sweeps")
