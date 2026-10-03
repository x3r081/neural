"""The novice setup path always selects the validated GPT-OSS model."""
from pathlib import Path

from .model_spec import GPTOSS_EXPECTED_REVISION

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_REPO = "openai/gpt-oss-120b"
DEFAULT_MODEL_REVISION = GPTOSS_EXPECTED_REVISION
DEFAULT_MODEL_LABEL = "GPT-OSS 120B"
DEFAULT_API_MODEL = "gpt-oss-120b-neural"


def default_paths(data_dir=None):
    data = Path(data_dir).expanduser().resolve() if data_dir is not None else ROOT / "data"
    return {
        "model_dir": data / "models" / "gpt-oss-120b",
        "raw_store_dir": data / "stores" / "gptoss120b_raw",
        "store_dir": data / "stores" / "gptoss120b_ps4",
    }
