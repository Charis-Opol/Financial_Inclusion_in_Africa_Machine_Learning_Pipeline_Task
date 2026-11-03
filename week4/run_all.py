"""The unified entry point for Week 4 experiments.

Run from the repo root, one stage at a time, reviewing results between stages:

  week4/.venv/Scripts/python.exe week4/run_all.py --stage 4a --device cuda
  week4/.venv/Scripts/python.exe week4/run_all.py --stage 4a --smoke     # minutes, on a tiny subset
  week4/.venv/Scripts/python.exe week4/run_all.py --stage e3 --arms mlp_dro --seeds 0 1

Finished folds are skipped, so rerunning the same command resumes after a dropped session.
Smoke runs write to week4/results/smoke/, so they can never mix with real results.

Not here: E1 (week4/src/e1_reproduce_baseline.py); E6 (SHAP), to be written after E2 and E3,
because it depends on which country is worst; E5c (H3), marked as future work (D1.8).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import tabpfn  # noqa: E402
import torch  # noqa: E402

from arms import ADJUSTMENT_CHOICES, build_arms  # noqa: E402
from harness import RunConfig, load_labelled, run_arm  # noqa: E402

SWEEP_ARMS = [f"mlp_dro_step_{s}" for s in (0.0, 0.001, 0.01, 0.1, 1.0)] + [f"mlp_dro_c_{c}" for c in ADJUSTMENT_CHOICES]

# Each experiment: (design, arms in run order). Sweep arms must follow their parent.
EXPERIMENTS: dict[str, tuple[str, list[str]]] = {
    "e2": ("nested", ["xgb", "xgb_spw", "mlp_erm", "tabpfn_v2"]),
    "e3": ("nested", ["mlp_erm_wc", "mlp_rw", "xgb_rw", "mlp_dro", *SWEEP_ARMS]),
    "e4": ("loco", ["xgb", "mlp_erm", "mlp_dro", "tabpfn_v2"]),
    "e5a": ("nested", ["h1_balanced"]),
    "e5b": ("nested", ["h2_erm", "h2_rw", "h2_dro"]),
    # E4 asks for "the best hybrids", so run only after E5, naming the hybrids with --arms.
    "e4_hybrids": ("loco", ["h1_balanced", "h2_rw", "h2_dro"]),
    "e7": ("nested", ["mlp_dro_country_label"]),
}
STAGES = {"4a": ["e2", "e3", "e4"], "4b": ["e5a", "e5b"], "4c": ["e7"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=[*STAGES, *EXPERIMENTS])
    parser.add_argument("--arms", nargs="*", help="Run only these arms of the chosen stage")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cpu", help="Device for TabPFN only, such as cpu or cuda")
    parser.add_argument("--smoke", action="store_true", help="Tiny subset and budgets: checks code paths only")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit(f"--device {args.device} was requested, but torch cannot see a GPU here.")
    config = RunConfig.smoke(args.device) if args.smoke else RunConfig(device=args.device)
    seeds = args.seeds[:1] if args.smoke else args.seeds

    frame = load_labelled(config)
    arms = build_arms()
    unknown = [name for name in args.arms or [] if name not in arms]
    if unknown:
        sys.exit(f"Unknown arms: {unknown}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "tabpfn": tabpfn.__version__,
                "torch": torch.__version__,
                "device": args.device,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "rows": len(frame),
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    for experiment in STAGES.get(args.stage, [args.stage]):
        design, names = EXPERIMENTS[experiment]
        for name in names:
            if args.arms and name not in args.arms:
                continue
            for seed in seeds:
                run_arm(arms[name], design, seed, frame, config)


if __name__ == "__main__":
    main()
