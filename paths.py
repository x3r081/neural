"""Machine-specific locations. Override with environment variables:
  NEURAL_MODEL_DIR  the downloaded Hugging Face checkpoint openai/gpt-oss-120b
  NEURAL_STORE_DIR  the expert store built from it by tools/build_store.py (~57 GiB), or its packed-scale form made by
                    tools/pack_store.py (~55 GiB, 2.88% smaller slots; the server reads the layout from the store)
  NEURAL_DEVKIT     optional: a MinGW-w64 gcc bin dir (only needed to rebuild the C kernels)
"""
import os

REPO = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("NEURAL_MODEL_DIR", r"G:\Models\gpt-oss-120b")
STORE_DIR = os.environ.get("NEURAL_STORE_DIR", r"G:\NeuralStores\gptoss120b_prepacked")
DEVKIT = os.environ.get("NEURAL_DEVKIT", r"F:\AI\Neural\third_party\tools\w64devkit\bin")
SAMPLE_CODE = os.path.join(REPO, "neural", "q80", "expert_runtime.py")   # a large source file used as test text


def add_devkit_dll_dir():
    if os.path.isdir(DEVKIT):
        os.add_dll_directory(DEVKIT)
