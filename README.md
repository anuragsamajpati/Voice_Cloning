# AKF Voice Cloning

This repository contains a proof-of-concept **Adaptive Kalman Filter (AKF)** based voice cloning / prosody tracking system converted from a Jupyter notebook into a standalone Python script, plus a small web dashboard for interactive testing.

The code is structured to follow a specific architecture:

- **HuBERT / MFCC feature extraction** (for research; realtime uses a faster MFCC path)
- **Learned deprojection `U`** and optional linear probe
- **Adaptive Kalman Filter** with learned noise models
- **Latent-to-mel decoder** + **PRN residual predictor** + mel smoothing
- **Griffin–Lim vocoder** to synthesize waveform

Realtime usage is focused on **short utterances (2–6 seconds)** due to CPU limitations.

---

## Files

- `AKF_base_algo_full.py`  
  Main script converted from the original notebook. Contains:
  - Data prep / synthetic example
  - Prosody extraction
  - Feature extraction (HuBERT + MFCC paths)
  - Linear probe and deprojection `U`
  - Adaptive Kalman Filter model and training
  - Mel decoder, PRN residual network, smoother, and vocoder
  - High-level function `run_voice_cloning_pipeline(...)` used for inference

- `akf_dashboard.py`  
  Gradio-based web dashboard with:
  - **Microphone** tab (record a short utterance)
  - **File Upload** tab (upload a WAV file)
  - Runs `run_voice_cloning_pipeline` in a **fast MFCC mode** for lower latency
  - Saves cloned output to `outputs/clone_*.wav`

- `akf_realtime_ui.py`  
  A simpler Gradio interface (mic → audio) using the same pipeline. The dashboard is the recommended entry point.

- `outputs/`  
  Created at runtime; stores generated WAV files. This folder is ignored by git.

---

## Requirements

This project was developed and tested on **macOS** with **Python 3.10+**.

Install dependencies (recommend using a virtual environment):

```bash
cd "Voice Cloning"
python3 -m venv .venv
source .venv/bin/activate  # on macOS/Linux

python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install librosa matplotlib numpy gradio soundfile transformers jiwer resemblyzer whisper scikit-learn pandas
```

You may adjust the PyTorch install command depending on your platform.

---

## Running the AKF realtime dashboard

1. Make sure you are in the project directory:

   ```bash
   cd "/Users/anuragsamajpati/Desktop/Voice Cloning"
   ```

2. Start the Gradio dashboard:

   ```bash
   python3 akf_dashboard.py
   ```

3. Wait until you see a line like:

   ```text
   Running on local URL:  http://127.0.0.1:7860
   ```

4. Open the printed URL in your browser (e.g. `http://127.0.0.1:7860`).

5. In the **Microphone** tab:
   - Click the mic and speak for **2–6 seconds**.
   - Stop the recording so you see a waveform.
   - Click **Clone from Mic**.
   - After a short delay, play **Cloned output** and check the saved file path.

6. In the **File Upload** tab:
   - Upload a short WAV (2–6 seconds).
   - Click **Clone from File**.
   - Listen to the cloned audio and check the saved file path.

> Note: The current implementation is CPU-bound and best suited for short utterances. Longer clips will take proportionally longer to process.

---

## Notes

- Several evaluation blocks (jiwer, Resemblyzer, Whisper, pitch correlation) are present in `AKF_base_algo_full.py` but are guarded so they only run when the file is executed directly (not when imported by the dashboard).
- The realtime dashboard uses `use_fast_features=True` in `run_voice_cloning_pipeline`, which replaces HuBERT with an MFCC+SVD path for quicker inference.
- The architecture code is intended for experimentation and educational purposes, not production deployment.
