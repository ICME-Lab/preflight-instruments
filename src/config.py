"""Central configuration."""
from pathlib import Path

# Any open-weights HF model works. Qwen2.5-1.5B-Instruct is small enough to run
# on a modest GPU (or CPU, slowly) while still having useful representations.
# Swap for a larger Qwen (7B/14B) for better probe quality.
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Which hidden layer to probe. Middle-to-late layers usually encode the most
# useful semantics. -1 would be the final layer; we default to a middle-late one.
# This index is into hidden_states, where index 0 is the embedding output, so a
# model with N transformer layers exposes N+1 hidden states (0..N).
LAYER_INDEX = 18

# Token pooling strategy for turning a (seq_len, hidden) matrix into one vector.
# "mean" is robust; "last" uses only the final token.
POOLING = "mean"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
ACTIVATIONS_PATH = DATA_DIR / "activations.npz"
PROBE_PATH = ARTIFACT_DIR / "probe.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

ARTIFACT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
