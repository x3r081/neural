# Quick start for Windows

Neural runs the official GPT-OSS 120B model on your PC. The checkpoint is about 61 GiB (65 GB); setup also builds two local expert stores. Allow about 195 GiB of free space on the drive used for the data directory. The installer checks capacity before it writes model files. Keep the PC online and do not close the installer while it is working.

## Before you begin

Use a 64-bit Windows 10 or 11 PC with Python 3.12 (64-bit), an NVIDIA GPU with BF16 support (Ampere or newer) and a driver that supports CUDA 13, a CPU with AVX2 and FMA, and an NVMe SSD with the free space above. The tested setup has an RTX 3080 Ti with 12 GiB of graphics memory and 64 GB of RAM; 64 GB is recommended, and the installer checks hardware compatibility on your PC. AVX-512 is used only with the verified native CPU build. Other compatible CPUs use the validated AVX2+FMA path, which is slower and has slower prompt processing; close memory-heavy programs while Neural runs.

Install Python 3.12 64-bit from [python.org](https://www.python.org/downloads/release/python-31210/) if needed. On that page, choose **Windows installer (64-bit)** and leave the **Python launcher** option selected in the installer. Neural checks for Python 3.12 64-bit and prints the official download link when it cannot find it. Neural creates a private `.venv` in its own folder, so it does not use another project's Python packages.

## Install and run

1. Download the repository from [GitHub](https://github.com/x3r081/neural) using **Code → Download ZIP**, extract the ZIP, and open the extracted `neural-main` folder. You can also use `git clone` if you already have Git.
2. Double-click `install_neural.bat`. The installer creates the private Python environment, downloads the pinned official GPT-OSS 120B checkpoint, builds the raw MXFP4 expert store, converts it to the lossless PS4 store, and verifies the result. This may take a while. Its default paths are `data\models\gpt-oss-120b`, `data\stores\gptoss120b_raw`, and `data\stores\gptoss120b_ps4`.
3. When installation finishes, double-click `start_neural.bat`. The same window opens a terminal chat and starts the local API at `http://127.0.0.1:8001/v1`.
4. Type a message after `you>` and press Enter. Type `/reset` to start a new conversation, `/effort low`, `/effort medium`, or `/effort high` to change reasoning effort, and `/quit` to close chat and stop the server started by this launcher.

To check an existing setup later, run `install_neural.bat -CheckOnly` from Command Prompt. This checks the configured model/store metadata, file sizes, and hardware plan without downloading or loading the model. It does not rehash every existing store payload.

The API listens on `127.0.0.1`, so it is available to programs on this PC. It is not exposed to other computers on your network. The terminal chat is the included user interface; the project site describes editor integrations for the separate NeuralServer runtime and those instructions do not configure this replacement.

## Choosing a data folder

By default, Neural stores the downloaded model and expert stores under the repository's `data` folder. To place these large files on a different drive, open Command Prompt in the repository folder and run:

```bat
install_neural.bat -DataDir D:\NeuralData
```

Use quotes if the path contains spaces, for example `install_neural.bat -DataDir "D:\AI Models\Neural Data"`. The installer checks available space at the chosen location. It keeps the model, raw store, and PS4 store after setup; it does not delete the raw files to recover space.

If `neural.local.json` already points to a GPT-OSS model and store whose metadata and file sizes validate, the installer reuses those paths. It will not replace a configuration that points to another model. See [`neural.local.example.json`](../neural.local.example.json) for the path format.

## If startup does not work

- **The installer asks for more disk space:** choose a drive with about 195 GiB free. The model and stores together use more than 170 GiB, and setup needs working room.
- **Setup stopped during raw-store creation and reports an incomplete store:** Neural preserves partial files and does not overwrite or delete them. Rerun with a new raw-store folder; for a custom data folder such as `D:\NeuralData`, run `install_neural.bat -DataDir D:\NeuralData -RawStoreDir D:\NeuralData\stores\gptoss120b_raw_retry`. The installer will reuse the valid checkpoint and other existing files at those paths.
- **The GPU or driver is unsupported:** update the NVIDIA driver and confirm that Windows detects the NVIDIA card. Neural's optimized GPT-OSS backend requires a CUDA-capable, BF16-capable GPU. The installer reports whether the available hardware plan is supported.
- **The terminal cannot connect to the API:** close other model servers using port 8001, then run `start_neural.bat` again. The launcher starts the API as part of the chat session.
- **Python is missing or the wrong version:** install Python 3.12 64-bit from the official link printed by the installer, then run the installer again.

The optional `start_neural_advisor.bat` opens a browser report about model fit and hardware capacity. The advisor uses catalog evidence and bandwidth estimates; it does not test answer quality and does not automatically change the default GPT-OSS model. See [model advisor](MODEL_ADVISOR.md).

The full architecture, benchmark scope, and known limitations are in the [technical overview](TECHNICAL_OVERVIEW.md). The [project site](https://x3r081.github.io/neural-site/) contains more historical detail about the standalone production runtime.
