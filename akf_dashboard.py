import os
import time
import numpy as np
import librosa
import soundfile as sf
import gradio as gr

from AKF_base_algo_full import (
    run_voice_cloning_pipeline,
    kf,
    U,
    cfg,
)

SR = cfg["sr"]
os.makedirs("outputs", exist_ok=True)


def clone_audio(audio):
    """Gradio callback: audio is (numpy_array, sample_rate)."""
    if audio is None:
        return None, "No audio provided."

    # Gradio may pass either just the array or (array, sr)
    if isinstance(audio, tuple) and len(audio) == 2:
        y_in, sr_in = audio
    else:
        y_in = audio
        sr_in = SR

    # Resample to model sample rate if needed
    if isinstance(sr_in, (int, float)) and sr_in != SR:
        y_in = librosa.resample(y_in, orig_sr=sr_in, target_sr=SR)
        sr_in = SR

    # Safety cap on maximum duration to keep processing responsive
    if isinstance(sr_in, (int, float)) and sr_in > 0:
        max_sec = 8.0
        max_samples = int(max_sec * sr_in)
        if y_in.shape[0] > max_samples:
            y_in = y_in[:max_samples]
            print(f"[clone_audio] Input truncated to {max_sec} seconds for faster processing.")

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
        use_fast_features=True,
    )

    # Save to a timestamped WAV file
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("outputs", f"clone_{ts}.wav")
    sf.write(out_path, waveform.astype(np.float32), sr_in)

    return (waveform.astype(np.float32), sr_in), f"Saved: {out_path}"


with gr.Blocks(title="AKF Voice Cloning Dashboard") as demo:
    # Custom HTML/CSS for a cleaner dashboard look
    gr.HTML(
        """
        <style>
        body { background: #050816; font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif; }
        .akf-container { max-width: 900px; margin: 0 auto; padding: 32px 16px 48px; }
        .akf-title { color: #e5e7eb; font-size: 28px; font-weight: 700; margin-bottom: 4px; }
        .akf-subtitle { color: #9ca3af; font-size: 14px; margin-bottom: 24px; }
        .akf-card { background: #0b1020; border-radius: 16px; padding: 20px 18px 24px; box-shadow: 0 18px 45px rgba(15,23,42,0.7); border: 1px solid #111827; }
        .akf-section-title { color: #e5e7eb; font-size: 16px; font-weight: 600; margin-bottom: 8px; }
        .akf-section-help { color: #9ca3af; font-size: 12px; margin-bottom: 12px; }
        button { border-radius: 9999px !important; font-weight: 600 !important; }
        button.primary { background: linear-gradient(90deg, #6366f1, #ec4899) !important; color: white !important; border: none !important; }
        button.primary:hover { filter: brightness(1.08); }
        .akf-audio .wrap { border-radius: 12px !important; overflow: hidden; }
        </style>
        """
    )

    with gr.Column(elem_classes=["akf-container"]):
        gr.Markdown("""<div class='akf-title'>AKF Voice Cloning</div>
<div class='akf-subtitle'>Speak or upload audio, then listen to the cloned output waveform.</div>""")

        with gr.Tab("Microphone"):
            with gr.Column(elem_classes=["akf-card"]):
                gr.Markdown(
                    """<div class='akf-section-title'>Realtime from Microphone</div>
<div class='akf-section-help'>Press the mic, speak a short sentence, then click <b>Clone from Mic</b> and wait for the model to process.</div>"""
                )
                mic_in = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Microphone input",
                    elem_classes=["akf-audio"],
                )
                mic_out = gr.Audio(
                    type="numpy",
                    label="Cloned output",
                    elem_classes=["akf-audio"],
                )
                mic_path = gr.Textbox(label="Saved file", interactive=False)
                mic_btn = gr.Button("Clone from Mic", elem_classes=["primary"])
                mic_btn.click(fn=clone_audio, inputs=mic_in, outputs=[mic_out, mic_path])

        with gr.Tab("File Upload"):
            with gr.Column(elem_classes=["akf-card"]):
                gr.Markdown(
                    """<div class='akf-section-title'>Clone from Existing Audio</div>
<div class='akf-section-help'>Upload a WAV file and click <b>Clone from File</b>. The system will resample and run the full AKF pipeline.</div>"""
                )
                file_in = gr.Audio(
                    sources=["upload"],
                    type="numpy",
                    label="Upload WAV",
                    elem_classes=["akf-audio"],
                )
                file_out = gr.Audio(
                    type="numpy",
                    label="Cloned output",
                    elem_classes=["akf-audio"],
                )
                file_path = gr.Textbox(label="Saved file", interactive=False)
                file_btn = gr.Button("Clone from File", elem_classes=["primary"])
                file_btn.click(fn=clone_audio, inputs=file_in, outputs=[file_out, file_path])


if __name__ == "__main__":
    demo.launch()
