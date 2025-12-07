import numpy as np
import gradio as gr
import librosa

from AKF_base_algo_full import (
    run_voice_cloning_pipeline,
    kf,
    U,
    cfg,
)

SR = cfg["sr"]


def clone_from_mic(audio):
    """Gradio callback: audio is (numpy_array, sample_rate)."""
    if audio is None:
        return None

    y_in, sr_in = audio
    if sr_in != SR:
        y_in = librosa.resample(y_in, orig_sr=sr_in, target_sr=SR)
        sr_in = SR

    # Run full AKF + U + mel/PRN + vocoder pipeline
    waveform, mel_db = run_voice_cloning_pipeline(
        y_in=y_in,
        sr=sr_in,
        kf_model=kf,
        U_model=U,
        probe_model=None,  # fall back to SVD-based projection if probe not available
        prosody_cfg=cfg,
        n_mels=80,
        history_steps=5,
    )

    # Gradio expects (waveform, sample_rate)
    return (waveform.astype(np.float32), sr_in)


demo = gr.Interface(
    fn=clone_from_mic,
    inputs=gr.Audio(source="microphone", type="numpy", label="Speak here"),
    outputs=gr.Audio(type="numpy", label="Cloned output"),
    title="AKF Voice Cloning (Mic → Audio)",
    description="Press the mic, speak a short phrase, then wait for the cloned audio.",
)


if __name__ == "__main__":
    demo.launch()
