"""Generate placeholder classifier weights for the dummy app/model.py.

Run once after cloning so the service has something to load at
models/classifier/dummy.pth without needing MODEL_DOWNLOAD_URL.
Predictions from these weights are random/meaningless — see private/README.md
to swap in the real architecture and real trained weights.

Deliberately NOT written to best_val_f1.pth / best_val_loss.pth / latest.pth —
those paths are reserved for real trained checkpoints and are exercised by
tests/test_real_model_weights.py (which correctly skips when they're absent).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from app.model import MultiScaleTCN

OUT_PATH = Path("models/classifier/dummy.pth")


def main():
    torch.manual_seed(0)
    model = MultiScaleTCN(num_classes=3)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, OUT_PATH)
    print(f"Wrote dummy checkpoint: {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
