# Converted from AKF_base_algo.ipynb
# Each section corresponds to one notebook cell.

# %% [code] 00_setup.py
# Basic imports and device selection
import os, random, math, time, json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# audio / features
import librosa

# optional: for HuBERT
# from transformers import Wav2Vec2FeatureExtractor, HubertModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# %% [code] 01_globals.py
# Simple global configuration you can edit
cfg = {
    "sr": 16000,
    "frame_ms": 20,
    "hop_ms": 20,
    "n_mfcc": 13,
    "state_dim": 16,
    "prosody_dim": 4,
    "kf_P0": 0.1,
}


def seconds_to_frames(sec, sr=cfg["sr"], hop_ms=cfg["hop_ms"]):
    hop = int(hop_ms * sr / 1000)
    return int(np.ceil(sec * sr / hop))


def frame_rate(sr=cfg["sr"], hop_ms=cfg["hop_ms"]):
    hop = int(hop_ms * sr / 1000)
    return sr / hop


# small plotting helper
def quick_plot_multi(lines, labels=None, title=None, figsize=(10, 3)):
    plt.figure(figsize=figsize)
    for arr in lines:
        plt.plot(arr)
    if labels:
        plt.legend(labels)
    if title:
        plt.title(title)
    plt.show()


# %% [code] 02_load_or_synth.py
# You can either load a wav (set wav_path) or create synthetic data for quick tests.

wav_path = "mic_test_8sec.wav"  # set to your file or None

if wav_path and os.path.exists(wav_path):
    y, sr = librosa.load(wav_path, sr=cfg["sr"])
    print("Loaded audio:", wav_path, "duration(s):", len(y) / sr)
else:
    # Synthetic example: sinusoidal slowly varying "true state" + noisy observations
    T = 400
    t = np.linspace(0, 6 * np.pi, T)
    state_dim = cfg["state_dim"]
    true_state = np.sin(t)[:, None] * (np.abs(np.random.randn(state_dim)) * 0.5)
    true_state = true_state + np.random.randn(*true_state.shape) * 0.03
    # prosody synthetic
    prosody = np.stack(
        [
            np.sin(t),
            np.cos(t * 0.5),
            1 / (1 + np.exp(-np.sin(t * 0.3))),
            np.ones_like(t),
        ],
        axis=1,
    )
    # obs = true_state + noise scale depending on prosody
    noise_scale = (1.2 - 0.8 * np.abs(prosody[:, 0]))[:, None]
    z_sim = true_state + np.random.randn(*true_state.shape) * noise_scale
    # put into torch tensors for later cells if using synthetic
    y, sr = None, cfg["sr"]
    print("Created synthetic data: T", T)


# %% [code] 03_prosody.py
# Use librosa to compute per-frame features (F0 via pyin, RMS, voiced)
assert (y is not None) or ("z_sim" in globals()), "Run cell 02 first."

frame_len = int(cfg["frame_ms"] * cfg["sr"] / 1000)
hop_len = int(cfg["hop_ms"] * cfg["sr"] / 1000)

if y is not None:
    # f0 + voiced probs
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=cfg["sr"],
        frame_length=frame_len,
        hop_length=hop_len,
        center=False,
    )
    if f0 is None:
        f0 = np.zeros(1)
        voiced_probs = np.zeros_like(f0)
    f0 = np.nan_to_num(f0, 0.0)
    d_f0 = np.concatenate([[0.0], np.diff(f0)]) if len(f0) > 1 else np.array([0.0])
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len, center=False)[0]
    voiced = (
        voiced_probs
        if voiced_probs is not None
        else (voiced_flag.astype(float) if voiced_flag is not None else np.zeros_like(f0))
    )
    # align lengths
    min_len = min(len(f0), len(d_f0), len(rms), len(voiced))
    f0, d_f0, rms, voiced = (
        f0[:min_len],
        d_f0[:min_len],
        rms[:min_len],
        voiced[:min_len],
    )
    # normalize
    prosody = np.stack([f0 / 300.0, d_f0 / 100.0, rms / (rms.max() + 1e-8), voiced], axis=1)
else:
    # If synthetic
    prosody = prosody  # from synthetic above

# convert to torch
prosody_t = torch.tensor(prosody, dtype=torch.float32, device=device)
n_frames = prosody.shape[0]
print("prosody shape:", prosody.shape, "frames:", n_frames)


# %% [code] 04_hubert_extract.py  (optional)
# This cell is kept for backwards compatibility but is no longer required for
# inference, since run_voice_cloning_pipeline computes HuBERT features directly
# from the provided input audio. We only run this block if real audio y exists.

from transformers import Wav2Vec2FeatureExtractor, HubertModel

# load model (takes time + memory)
feat = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
model.eval()

if y is not None:
    inputs = feat(y, sampling_rate=cfg["sr"], return_tensors="pt", padding="longest")
    with torch.no_grad():
        out = model(inputs["input_values"].to(device))
    hidden = out.last_hidden_state[0].cpu().numpy()  # (T_model, D)
    # resample/interpolate to n_frames (same strategy used earlier)
    T_model, feat_dim = hidden.shape
    t_model = np.linspace(0, 1, T_model)
    t_target = np.linspace(0, 1, n_frames)
    hidden_resampled = np.stack(
        [np.interp(t_target, t_model, hidden[:, d]) for d in range(feat_dim)], axis=1
    )
    print("HuBERT resampled shape:", hidden_resampled.shape)

    # project to state_dim with SVD
    Zc = hidden_resampled - hidden_resampled.mean(axis=0, keepdims=True)
    U_svd, Svals, Vt = np.linalg.svd(Zc, full_matrices=False)
    Vtop = Vt[: cfg["state_dim"], :]
    z_hubert_proj = Zc @ Vtop.T
    # normalize same way later code expects
    z_mean = z_hubert_proj.mean(axis=0, keepdims=True)
    z_std = z_hubert_proj.std(axis=0, keepdims=True) + 1e-9
    z_hubert_norm = (z_hubert_proj - z_mean) / z_std
    z_t = torch.tensor(z_hubert_norm, dtype=torch.float32, device=device)
    print("z_t (HuBERT->proj) shape:", z_t.shape)
else:
    print("[HuBERT import cell] No real audio y; skipping HuBERT feature precompute (inference will compute features per call).")


# %% [code] Cell A: Learnable linear projection from HuBERT embeddings -> state space
import torch.nn.functional as F  # ensure F is imported
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# If no HuBERT features were precomputed (e.g., synthetic run), skip this cell.
if "hidden_resampled" not in globals():
    print("[LinearProbe] hidden_resampled not found; skipping probe pretraining (realtime pipeline computes features per call).")
else:
    # Assume you already have `hidden_resampled` from HuBERT (n_frames, feat_dim)
    # and `Zn_tensor` as normalized target proxy (n_frames, state_dim)
    feat_dim = hidden_resampled.shape[1]
    state_dim = cfg["state_dim"]

    class LinearProbe(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.proj = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            return self.proj(x)

    probe = LinearProbe(feat_dim, state_dim).to(device)

    # Prepare tensors
    X = torch.tensor(hidden_resampled, dtype=torch.float32, device=device)
    # Here we use z_hubert_norm as proxy target; notebook later rebuilds Zn
    Y = torch.tensor(z_hubert_norm, dtype=torch.float32, device=device)
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    # Pretrain probe with simple MSE
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-5)
    n_epochs = 30
    for epoch in range(n_epochs):
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            pred = probe(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        print(f"Epoch {epoch+1}/{n_epochs} loss {total/len(ds):.6f}")

    # Replace z_t by learned projection output
    with torch.no_grad():
        z_proj_learned = probe(X).cpu().numpy()

    # Normalize same way as before
    z_mean = z_proj_learned.mean(axis=0, keepdims=True)
    z_std = z_proj_learned.std(axis=0, keepdims=True) + 1e-9
    z_learned_norm = (z_proj_learned - z_mean) / z_std
    z_t = torch.tensor(z_learned_norm, dtype=torch.float32, device=device)

    print("Learnable probe pretrain done. z_t shape:", z_t.shape)


# %% [code] force recompute by removing old cached variables
for v in ["z_t", "Zn", "Zn_tensor", "Z_warp", "Z_orig"]:
    if v in globals():
        del globals()[v]
print("Deleted cached z_t/Zn if present. Now re-run Cell 05.")


# %% [code] FIX: rebuild z_t and Zn with the current cfg["state_dim"]
import numpy as np
import torch

state_dim = cfg["state_dim"]
print("Desired state_dim:", state_dim)

# --- pick source features: prefer HuBERT resampled if available, else MFCC 'z' array ---

# Case A: If HuBERT resampled matrix exists from Cell 04 (hidden_resampled)
if "hidden_resampled" in globals():
    print("Using HuBERT resampled features (hidden_resampled).")
    features = hidden_resampled.copy()  # numpy (n_frames, feat_dim)

# Case B: else if MFCC 'z' exists (from earlier), use that
elif "z" in globals():
    print("Using MFCC matrix 'z' (mfcc.T).")
    features = z.copy()  # (n_frames_mfcc, n_mfcc)

# Case C: else, if you used synthetic z_sim, use that
elif "z_sim" in globals():
    print("Using synthetic z_sim.")
    features = z_sim.copy()
else:
    raise RuntimeError(
        "No feature matrix found (hidden_resampled, z, or z_sim). Run Cell 04 or Cell 05 feature extraction first."
    )

# Ensure features length >= n_frames; align to prosody / n_frames if needed
n_frames_target = prosody.shape[0] if "prosody" in globals() else features.shape[0]
if features.shape[0] >= n_frames_target:
    features = features[:n_frames_target, :]
else:
    # pad last row to match length
    pad_count = n_frames_target - features.shape[0]
    last = np.repeat(features[-1:, :], pad_count, axis=0)
    features = np.vstack([features, last])

print("features.shape:", features.shape)

# 1) center features, compute SVD, take top state_dim components
Zc = features - features.mean(axis=0, keepdims=True)
U_svd, Svals, Vt = np.linalg.svd(Zc, full_matrices=False)
if state_dim > Vt.shape[0]:
    raise ValueError(
        f"Requested state_dim {state_dim} is larger than available feature dim {Vt.shape[0]}. Reduce state_dim or use higher-dim features."
    )
Vtop = Vt[:state_dim, :]  # (state_dim, feat_dim)
z_proj = Zc @ Vtop.T  # (n_frames_target, state_dim)

# 2) normalize projected features (mean/std)
mean_z = z_proj.mean(axis=0, keepdims=True)
std_z = z_proj.std(axis=0, keepdims=True) + 1e-9
z_norm = (z_proj - mean_z) / std_z

# 3) set global z_t and Zn variables used by later cells
z_t = torch.tensor(z_norm, dtype=torch.float32, device=device)
Zn = z_norm.copy()  # numpy copy as target proxy
Zn_tensor = torch.tensor(Zn, dtype=torch.float32, device=device)

print("Rebuilt z_t and Zn with state_dim:", state_dim)
print("z_t shape:", z_t.shape, "Zn_tensor shape:", Zn_tensor.shape)


# %% [code] 05_build_z.py
# If you didn't run HuBERT, fallback to MFCC projection used earlier
if "z_t" not in globals():
    if y is not None:
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=cfg["sr"],
            n_mfcc=cfg["n_mfcc"],
            n_fft=frame_len,
            hop_length=hop_len,
            center=False,
        )
        z = mfcc.T  # (n_frames_mfcc, n_mfcc)
        # align lengths
        if z.shape[0] >= n_frames:
            z = z[:n_frames]
        else:
            pad_count = n_frames - z.shape[0]
            z = np.vstack([z, np.repeat(z[-1:, :], pad_count, axis=0)])
        # project to state_dim using SVD
        Zc = z - z.mean(axis=0, keepdims=True)
        U_svd, Svals, Vt = np.linalg.svd(Zc, full_matrices=False)
        Vtop = Vt[: cfg["state_dim"], :]
        z_proj = Zc @ Vtop.T
        # normalize and set z_t
        mean_z = z_proj.mean(axis=0, keepdims=True)
        std_z = z_proj.std(axis=0, keepdims=True) + 1e-9
        z_norm = (z_proj - mean_z) / std_z
        z_t = torch.tensor(z_norm, dtype=torch.float32, device=device)
    elif "z_sim" in globals():
        z_t = torch.tensor(z_sim, dtype=torch.float32, device=device)
    else:
        raise RuntimeError("No z_t candidate found. Run HuBERT or MFCC extraction.")
print("z_t shape:", z_t.shape)

# Build normalized target Zn (for synthetic or for debug target)
# If synthetic true_state exists use it, else set Zn := z_t (proxy target)
if "true_state" in globals():
    Zn = (true_state - true_state.mean(axis=0, keepdims=True)) / (
        true_state.std(axis=0, keepdims=True) + 1e-9
    )
    Zn_tensor = torch.tensor(Zn, dtype=torch.float32, device=device)
else:
    # When using unlabeled real data we treat z_norm as "target/latent proxy"
    Zn = z_t.cpu().numpy()
    Zn_tensor = z_t.clone()
print("Zn shape:", Zn_tensor.shape)


# %% [code] 06_models.py
class AdaptiveKalman(nn.Module):
    def __init__(self, state_dim, prosody_dim):
        super().__init__()
        self.state_dim = state_dim
        self.mlp_q = nn.Sequential(
            nn.Linear(prosody_dim, 64), nn.ReLU(), nn.Linear(64, state_dim)
        )
        self.mlp_r = nn.Sequential(
            nn.Linear(prosody_dim, 64), nn.ReLU(), nn.Linear(64, state_dim)
        )
        self.gamma_head = nn.Sequential(
            nn.Linear(prosody_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward_step(self, mu, P, z_t, p_t):
        Q_t = F.softplus(self.mlp_q(p_t)) + 1e-8
        R_t = F.softplus(self.mlp_r(p_t)) + 1e-8
        P_pred = P + Q_t
        K = P_pred / (P_pred + R_t)
        K = torch.clamp(K, min=1e-6, max=0.95)
        gamma_t = torch.sigmoid(self.gamma_head(p_t)).squeeze(-1)
        mu_new = mu + gamma_t * (K * (z_t - mu))
        P_new = (1 - K) * P_pred
        return mu_new, P_new, Q_t, R_t, K, gamma_t


class LinearDeproj(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(dim))

    def forward(self, x):
        return self.lin(x)


class ResidualMLPDeproj(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        return x + 0.1 * self.net(x)


# %% [code] 07_helpers.py

def compute_full_sequence_stats(
    kf_module, U_module, Z_warp_tensor, prosody_tensor, Zn_tensor, P0=cfg["kf_P0"]
):
    with torch.no_grad():
        mu = torch.zeros(cfg["state_dim"], device=device)
        P = torch.ones(cfg["state_dim"], device=device) * P0
        mse_sum = 0.0
        Ks, Gammas = [], []
        mus = []
        for t in range(Zn_tensor.shape[0]):
            z_in = Z_warp_tensor[t]
            z_lat = U_module(z_in)
            out = kf_module.forward_step(mu, P, z_lat, prosody_tensor[t])
            mu, P, Q_t, R_t, K, gamma = out
            mse_sum += ((mu - Zn_tensor[t]) ** 2).mean().item()
            Ks.append(K.mean().item())
            Gammas.append(float(gamma))
            mus.append(mu.cpu().numpy())
    return mse_sum / Zn_tensor.shape[0], np.mean(Ks), np.mean(Gammas), np.array(mus)


print("Zn_np shape:", Zn_tensor.shape)
print("cfg state_dim:", cfg["state_dim"])


# %% [code] 08_pretrain_U.py
# Create warp A_lang (orthonormal) and mask for middle third; train linear U with UA reg
state_dim = cfg["state_dim"]
# create warp
W = np.random.RandomState(42).randn(state_dim, state_dim)
Q_mat, _ = np.linalg.qr(W)
A_lang = Q_mat.astype(np.float32)
A_lang_torch = torch.tensor(A_lang, dtype=torch.float32, device=device)

# normalized input matrix Zn (numpy) exists; use Z_warp (normalized space)
Zn_np = Zn if isinstance(Zn, np.ndarray) else Zn_tensor.cpu().numpy()
z_warp_np = Zn_np.copy()
mask = np.zeros(Zn_np.shape[0], dtype=bool)
mask[Zn_np.shape[0] // 3 : 2 * Zn_np.shape[0] // 3] = True
z_warp_np[mask] = Zn_np[mask] @ A_lang.T  # warp region

Z_orig = torch.tensor(Zn_np, dtype=torch.float32, device=device)
Z_warp = torch.tensor(z_warp_np, dtype=torch.float32, device=device)

# instantiate U linear and init with pinv(A)
U = LinearDeproj(state_dim).to(device)
pinvA = np.linalg.pinv(A_lang).astype(np.float32)
with torch.no_grad():
    U.lin.weight.copy_(torch.from_numpy(pinvA).to(device))

opt = torch.optim.Adam(U.lin.parameters(), lr=5e-4, weight_decay=1e-6)
lambda_reg = 1e-3
losses = []
for epoch in range(300):
    opt.zero_grad()
    z_hat = U.lin(Z_warp)
    mse = (
        F.mse_loss(z_hat[mask], Z_orig[mask]) if mask.sum() else F.mse_loss(z_hat, Z_orig)
    )
    UA = U.lin.weight @ A_lang_torch
    frob_reg = torch.norm(UA - torch.eye(state_dim, device=device), p="fro")
    loss = mse + lambda_reg * frob_reg
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if epoch % 50 == 0:
        print("epoch", epoch, "loss", loss.item(), "mse", mse.item(), "frob", frob_reg.item())

print("Pretrain done. Final mse:", losses[-1])
# keep U, Z_warp, Z_orig, mask around for later stages


# %% [code] 09_train_kf_frozenU.py
kf = AdaptiveKalman(state_dim=cfg["state_dim"], prosody_dim=prosody_t.shape[1]).to(
    device
)
# freeze U
for p in U.parameters():
    p.requires_grad = False

opt_kf = torch.optim.Adam(kf.parameters(), lr=1e-3)
unroll_len = 80
T_total = Zn_tensor.shape[0]

for epoch in range(40):
    total_loss = 0.0
    count = 0
    for start in range(0, T_total - unroll_len, unroll_len):
        mu = torch.zeros(cfg["state_dim"], device=device)
        P = torch.ones(cfg["state_dim"], device=device) * cfg["kf_P0"]
        seq_loss = 0.0
        for t in range(unroll_len):
            idx = start + t
            z_in = Z_warp[idx]
            z_lat = U(z_in).detach()
            mu, P, Q_t, R_t, K, gamma = kf.forward_step(mu, P, z_lat, prosody_t[idx])
            seq_loss += ((mu - Zn_tensor[idx]) ** 2).mean()
        seq_loss = seq_loss / unroll_len
        opt_kf.zero_grad()
        seq_loss.backward()
        opt_kf.step()
        total_loss += seq_loss.item()
        count += 1
    print(f"epoch {epoch} avg loss {total_loss/max(1,count):.6f}")
print("KF pretraining done.")


# %% [code] 10_joint_finetune (simplified header)
import torch, numpy as np, random
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

# This joint finetune stage is only run if a LinearProbe exists. For realtime
# inference with synthetic data, probe may be undefined and we skip this block.
if "probe" not in globals():
    print("[JointFinetune] probe not defined; skipping joint finetune block.")
else:
    for p in probe.parameters():
        p.requires_grad = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert "U" in globals() and "kf" in globals(), "Ensure U and kf are defined"
    # normalized target (Zn) and observed Z_warp tensors must exist
    assert "Zn" in globals() or "Zn_tensor" in globals()
    if "Zn_tensor" not in globals():
        Zn_tensor = torch.tensor(Zn, dtype=torch.float32, device=device)
    else:
        Zn_tensor = Zn_tensor.to(device)

    Z_warp_tensor = Z_warp.to(device) if "Z_warp" in globals() else Zn_tensor.clone()
    prosody_tensor = prosody_t.to(device)

    T_total = Zn_tensor.shape[0]
    state_dim = Zn_tensor.shape[1]

    # Hyperparams
    unroll_len = 160
    n_epochs = 120
    P0_val = 1.0
    kf_lr = 5e-5
    U_lr = 5e-6
    weight_warp = 2.0
    mu_smooth_lambda = 1e-3
    grad_clip = 1.0

    # diagnostics history
    epoch_logs = []

    # Put modules to device
    U.to(device)
    kf.to(device)

    # optimizer with separate lrs
    opt = torch.optim.Adam(
        [
            {"params": kf.parameters(), "lr": 5e-5},
            {"params": U.parameters(), "lr": 5e-6},
            {"params": probe.parameters(), "lr": 1e-4},
        ],
        eps=1e-8,
    )
    sched = CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-7)

    # build window start indices and shuffle each epoch
    windows = [i for i in range(0, T_total - unroll_len, unroll_len)]
    print(f"Joint finetune start: epochs={n_epochs}, unroll_len={unroll_len}, #windows={len(windows)}")

    for epoch in range(n_epochs):
        random.shuffle(windows)  # shuffle windows each epoch
        epoch_loss = 0.0
        epoch_mse = 0.0
        count = 0

        for start in windows:
            end = start + unroll_len
            # reset KF state per window
            mu = torch.zeros(state_dim, device=device)
            P = torch.ones(state_dim, device=device) * P0_val

            seq_loss = 0.0
            prev_mu = mu.clone()
            for t in range(unroll_len):
                idx = start + t
                z_in = Z_warp_tensor[idx]
                z_lat = U(z_in)  # deproject
                out = kf.forward_step(mu, P, z_lat, prosody_tensor[idx])
                if len(out) == 6:
                    mu, P, Q_t, R_t, K, gamma = out
                else:
                    mu, P, Q_t, R_t, K = out
                    gamma = torch.tensor(1.0, device=device)
                target = Zn_tensor[idx]

                # per-step MSE
                step_mse = ((mu - target) ** 2).mean()
                # weight if this frame is in the warped region (if `mask` exists in global scope)
                if "mask" in globals() and mask[idx]:
                    step_mse = step_mse * weight_warp
                # accumulate
                seq_loss = seq_loss + step_mse

                # smoothness penalty (small): encourage small change in mu between steps
                seq_loss = seq_loss + mu_smooth_lambda * ((mu - prev_mu) ** 2).mean()
                prev_mu = mu

            seq_loss = seq_loss / unroll_len

            opt.zero_grad()
            seq_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(kf.parameters()) + list(U.parameters()), max_norm=grad_clip)
            opt.step()

            epoch_loss += seq_loss.item()
            epoch_mse += seq_loss.item()
            count += 1

        sched.step()
        avg_loss = epoch_loss / max(1, count)
        avg_mse = epoch_mse / max(1, count)
        # compute light diagnostics over full seq (no grad)
        with torch.no_grad():
            mu_diag = torch.zeros(state_dim, device=device)
            P_diag = torch.ones(state_dim, device=device) * P0_val
            Ks, Gammas = [], []
            for t in range(T_total):
                z_in = probe(torch.tensor(hidden_resampled[t], dtype=torch.float32, device=device))
                z_lat = U(z_in)
                out = kf.forward_step(mu_diag, P_diag, z_lat, prosody_tensor[t])
                if len(out) == 6:
                    mu_diag, P_diag, Qd, Rd, Kd, gammad = out
                else:
                    mu_diag, P_diag, Qd, Rd, Kd = out
                    gammad = torch.tensor(1.0, device=device)
                Ks.append(Kd.mean().item())
                Gammas.append(float(gammad))
            meanK = float(np.mean(Ks))
            meanGamma = float(np.mean(Gammas))

        epoch_logs.append((avg_loss, avg_mse, meanK, meanGamma))
        print(
            f"Epoch {epoch+1}/{n_epochs} avg_loss {avg_loss:.6f} avg_mse {avg_mse:.6f} meanK {meanK:.4f} meanGamma {meanGamma:.4f}"
        )

    print("Joint finetune finished. Saving small diagnostics to 'epoch_logs' variable.")
    loss_vals = [x[0] for x in epoch_logs]
    plt.figure(figsize=(6, 3))
    plt.plot(loss_vals)
    plt.title("Joint finetune avg loss per epoch")
    plt.show()


# %% [code] 11_eval.py
if __name__ == "__main__":
    mse, meanK, meanGamma, mus = compute_full_sequence_stats(
        kf, U, Z_warp, prosody_t, Zn_tensor, P0=1.0
    )
    print("Final tracking MSE:", mse, "meanK:", meanK, "meanGamma:", meanGamma)

    # plot first component
    plt.figure(figsize=(9, 3))
    plt.plot(mus[:, 0], label="mu[0]")
    plt.plot(Zn_tensor.cpu().numpy()[:, 0], label="target Zn[0]", alpha=0.7)
    plt.legend()
    plt.title("mu vs Zn (comp0)")
    plt.show()


# %% [code] 12_checkpoints.py

def save_ckpt(prefix="akf"):
    torch.save({"U": U.state_dict(), "kf": kf.state_dict()}, f"{prefix}_ckpt.pth")
    print("Saved ckpt:", f"{prefix}_ckpt.pth")


def load_ckpt(path):
    d = torch.load(path, map_location=device)
    U.load_state_dict(d["U"])
    kf.load_state_dict(d["kf"])
    print("Loaded ckpt:", path)


if __name__ == "__main__":
    # %% [code] 13_eval_setup_jiwer
    # ! pip install jiwer
    import jiwer
    import soundfile as sf
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import roc_curve
    from scipy.stats import pearsonr
    import pandas as pd

    # Paths to evaluation WAVs (edit as needed)
    EVAL_FILES = {
        "baseline": "baseline_true_gl.wav",
        "prn_res": "prn_pred_res_gl.wav",
        "prn_abs": "prn_pred_abs_gl.wav",  # optional
    }

    print("Evaluation files:", EVAL_FILES)

    # %% [code] 14_install_resemblyzer_whisper
    # The original notebook used shell pip commands; keep as comments here.
    # ! pip install resemblyzer soundfile whisper tabulate

    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder()  # loads model

    # %% [code] 15_speaker_embeddings
    import os

    def embed_audio_resemblyzer(path):
        # returns 1-D numpy embedding
        wav, sr = sf.read(path)
        # Resemblyzer's preprocess_wav expects either path or ndarray + sr:
        wav_proc = preprocess_wav(path)  # convenience function that loads & resamples
        emb = encoder.embed_utterance(wav_proc)  # (d,)
        return emb

    # try to embed evaluation files
    EVAL_FILES = {
        "baseline": "baseline_true_gl.wav",
        "prn_res": "prn_pred_res_gl.wav",
        # "prn_abs": "prn_pred_abs_gl.wav",  # optional
    }
    embs = {}
    for name, p in EVAL_FILES.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found; generate it before running evaluation.")
        embs[name] = embed_audio_resemblyzer(p)
    print("Extracted embeddings for:", list(embs.keys()))

    # %% [code] 16_cosine_similarity
    from sklearn.metrics.pairwise import cosine_similarity

    def cos_sim(a, b):
        return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])

    cos_baseline_prn = cos_sim(embs["baseline"], embs["prn_res"])
    print("Cosine similarity (baseline vs prn_res):", cos_baseline_prn)

    # %% [code] 17_quick_eer
    import numpy as np
    from sklearn.metrics import roc_curve

    # Create a simple impostor by reversing baseline audio (guaranteed different speaker content)
    y, sr = sf.read(EVAL_FILES["baseline"])
    impostor_path = "impostor_temp.wav"
    sf.write(impostor_path, y[::-1], sr)

    emb_impostor = embed_audio_resemblyzer(impostor_path)

    # Prepare scores and labels (1 = genuine, 0 = impostor)
    scores = np.array(
        [
            cos_sim(embs["baseline"], embs["prn_res"]),
            cos_sim(embs["baseline"], emb_impostor),
        ]
    )
    labels = np.array([1, 0])

    # Compute ROC and EER (very small sample; this is a quick check)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    print("Quick EER estimate (toy):", eer)

    # NOTE: For real EER report, create many genuine/impostor pairs across speakers.

    # %% [code] 18_install_openai_whisper
    # !pip uninstall -y whisper
    # !pip install -U openai-whisper

    import whisper

    # %% [code] 19_asr_wer
    # ASR (Whisper) WER/CER

    # load whisper model once
    asr = whisper.load_model("small")  # or "base" if you prefer smaller

    def transcribe_whisper(path):
        r = asr.transcribe(path)
        return r["text"].strip()

    # You must fill in the reference transcript for this audio (ground truth)
    # If you have many eval files, provide a dict mapping filenames->transcripts
    GROUND_TRUTH = "<PASTE_GROUND_TRUTH_TRANSCRIPT_HERE>"

    hyp_prn_res = transcribe_whisper(EVAL_FILES["prn_res"])
    print("Hypothesis (prn_res):", hyp_prn_res)
    wer_prn_res = jiwer.wer(GROUND_TRUTH, hyp_prn_res)
    print("WER (prn_res):", wer_prn_res)

    # %% [code] 20_pitch_correlation
    import librosa
    from scipy.stats import pearsonr

    def extract_f0(path, sr=16000, hop_length=320):
        y, _ = librosa.load(path, sr=sr)
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=500, sr=sr, hop_length=hop_length
        )
        return f0, voiced_flag

    f0_ref, v_ref = extract_f0(EVAL_FILES["baseline"])
    f0_gen, v_gen = extract_f0(EVAL_FILES["prn_res"])

    # mask voiced frames present in both
    mask_f0 = (~np.isnan(f0_ref)) & (~np.isnan(f0_gen))
    if mask_f0.sum() < 10:
        print("Warning: too few voiced frames for reliable correlation.")
        r = None
    else:
        r, p = pearsonr(f0_ref[mask_f0], f0_gen[mask_f0])
        print("F0 Pearson r:", r, "p-value:", p)

    # %% [code] 21_assemble_results
    rows = {
        "system": ["AKF_prn_res"],
        "cosine_baseline_prnres": [cos_baseline_prn],
        "eer_toy": [eer],
        "wer_prn_res": [wer_prn_res],
        "f0_corr": [r if "r" in globals() else None],
    }

    df = pd.DataFrame(rows)
    df.to_csv("akf_eval_results.csv", index=False)
    print("Saved akf_eval_results.csv")
    print(df)


# %% [code] 22_mel_models_and_vocoder
# Mel decoder, PRN residual predictor, mel normalization & smoothing, and vocoder wrapper

import librosa


class LatentToMel(nn.Module):
    """Baseline latent-to-mel decoder.

    Maps latent state (mu_t) -> mel frame.
    """

    def __init__(self, state_dim: int, n_mels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_mels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, state_dim) or (B, state_dim)
        return self.net(x)


class ResidualMelPRN(nn.Module):
    """PRN residual predictor using latent state, prosody, and mel history.

    Inputs per step: latent_t, prosody_t, mel_hist (T_hist x n_mels).
    """

    def __init__(self, state_dim: int, prosody_dim: int, n_mels: int, history_steps: int = 5):
        super().__init__()
        self.history_steps = history_steps
        in_dim = state_dim + prosody_dim + n_mels * history_steps
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_mels),
        )

    def forward(
        self,
        latent_t: torch.Tensor,
        prosody_t: torch.Tensor,
        mel_hist: torch.Tensor,
    ) -> torch.Tensor:
        # latent_t: (B, state_dim)
        # prosody_t: (B, prosody_dim)
        # mel_hist: (B, history_steps, n_mels)
        B, H, M = mel_hist.shape
        mel_flat = mel_hist.reshape(B, H * M)
        x = torch.cat([latent_t, prosody_t, mel_flat], dim=-1)
        return self.net(x)


class MelSmoother(nn.Module):
    """Simple temporal mel smoother (1D conv over time)."""

    def __init__(self, n_mels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(n_mels, n_mels, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, mel_seq: torch.Tensor) -> torch.Tensor:
        # mel_seq: (T, n_mels)
        x = mel_seq.T.unsqueeze(0)  # (1, n_mels, T)
        y = self.conv(x)
        return y.squeeze(0).T  # (T, n_mels)


mel_norm_stats: dict = {}


def compute_mel_norm_stats(mel_tensor: torch.Tensor):
    """Compute and cache per-bin mean/std for mel normalization."""

    global mel_norm_stats
    mean = mel_tensor.mean(dim=0, keepdim=True)
    std = mel_tensor.std(dim=0, keepdim=True) + 1e-6
    mel_norm_stats = {"mean": mean, "std": std}
    return mean, std


def normalize_mel(mel_tensor: torch.Tensor) -> torch.Tensor:
    global mel_norm_stats
    if not mel_norm_stats:
        compute_mel_norm_stats(mel_tensor)
    mean, std = mel_norm_stats["mean"], mel_norm_stats["std"]
    return (mel_tensor - mean) / std


def denormalize_mel(mel_norm_tensor: torch.Tensor) -> torch.Tensor:
    global mel_norm_stats
    assert mel_norm_stats, "mel_norm_stats not computed yet. Call normalize_mel on training data first."
    mean, std = mel_norm_stats["mean"], mel_norm_stats["std"]
    return mel_norm_tensor * std + mean


class GriffinLimVocoder:
    """Simple Griffin-Lim vocoder using librosa.

    Expects mel in dB scale or linear mel depending on configuration.
    """

    def __init__(self, sr: int = 16000, n_fft: int = 1024, hop_length: int | None = None, n_mels: int = 80):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else int(cfg["hop_ms"] * sr / 1000)
        self.n_mels = n_mels

    def __call__(self, mel_db: np.ndarray) -> np.ndarray:
        """mel_db: (T, n_mels), dB-scaled mel spectrogram.

        Returns waveform (1D np.ndarray).
        """

        # Transpose to (n_mels, T)
        mel_db_T = mel_db.T
        # Convert dB -> power
        mel_power = librosa.db_to_power(mel_db_T)
        # Invert mel to linear spectrogram
        S = librosa.feature.inverse.mel_to_stft(
            mel_power,
            sr=self.sr,
            n_fft=self.n_fft,
        )
        # Griffin-Lim to waveform
        y = librosa.griffinlim(S, n_iter=60, hop_length=self.hop_length, n_fft=self.n_fft)
        return y


# %% [code] 23_decode_and_inference
def decode_mel_from_latent(
    latent_seq: torch.Tensor,
    prosody_seq: torch.Tensor,
    base_decoder: LatentToMel,
    prn: ResidualMelPRN,
    smoother: MelSmoother | None = None,
    history_steps: int = 5,
) -> torch.Tensor:
    """Run baseline mel + PRN residual + (optional) smoothing.

    latent_seq: (T, state_dim)
    prosody_seq: (T, prosody_dim)
    Returns: mel_final (T, n_mels) in *normalized mel space* (use denormalize_mel for dB).
    """

    device_local = latent_seq.device
    T, state_dim_local = latent_seq.shape

    # Baseline mel from latent
    mel_base = base_decoder(latent_seq)  # (T, n_mels)
    n_mels = mel_base.shape[1]

    # Initialize mel history with zeros
    hist = torch.zeros(history_steps, n_mels, device=device_local)
    mel_out = []

    for t in range(T):
        # Build history tensor (1, H, M)
        mel_hist = hist.unsqueeze(0)
        latent_t = latent_seq[t].unsqueeze(0)  # (1, state_dim)
        prosody_t = prosody_seq[t].unsqueeze(0)  # (1, prosody_dim)
        res_t = prn(latent_t, prosody_t, mel_hist)  # (1, n_mels)
        mel_t = mel_base[t].unsqueeze(0) + res_t  # residual combination
        mel_out.append(mel_t.squeeze(0))

        # Update history (FIFO)
        hist = torch.roll(hist, shifts=-1, dims=0)
        hist[-1] = mel_t.squeeze(0).detach()

    mel_seq = torch.stack(mel_out, dim=0)  # (T, n_mels)

    if smoother is not None:
        mel_seq = smoother(mel_seq)

    return mel_seq


def run_voice_cloning_pipeline(
    y_in: np.ndarray,
    sr: int,
    kf_model: AdaptiveKalman,
    U_model: nn.Module,
    probe_model: nn.Module | None,
    prosody_cfg: dict | None = None,
    n_mels: int = 80,
    history_steps: int = 5,
    use_fast_features: bool = False,
) -> tuple[np.ndarray, torch.Tensor]:
    """High-level pipeline following the architecture diagram for arbitrary input audio.

    Steps:
      1) Prosody extraction from y_in.
      2) HuBERT -> probe -> learned deprojection U -> latent sequence via adaptive KF.
      3) Latent -> mel via baseline decoder + PRN residual + smoothing.
      4) Mel denorm & Griffin-Lim vocoder.

    Returns (waveform, mel_final_db).
    """

    if prosody_cfg is None:
        prosody_cfg = cfg

    # --- 1) Prosody extraction from y_in ---
    frame_len_loc = int(prosody_cfg["frame_ms"] * sr / 1000)
    hop_len_loc = int(prosody_cfg["hop_ms"] * sr / 1000)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y_in,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=frame_len_loc,
        hop_length=hop_len_loc,
        center=False,
    )
    if f0 is None:
        f0 = np.zeros(1)
        voiced_probs = np.zeros_like(f0)
    f0 = np.nan_to_num(f0, 0.0)
    d_f0 = np.concatenate([[0.0], np.diff(f0)]) if len(f0) > 1 else np.array([0.0])
    rms = librosa.feature.rms(
        y=y_in, frame_length=frame_len_loc, hop_length=hop_len_loc, center=False
    )[0]
    voiced = (
        voiced_probs
        if voiced_probs is not None
        else (voiced_flag.astype(float) if voiced_flag is not None else np.zeros_like(f0))
    )
    min_len_loc = min(len(f0), len(d_f0), len(rms), len(voiced))
    f0, d_f0, rms, voiced = (
        f0[:min_len_loc],
        d_f0[:min_len_loc],
        rms[:min_len_loc],
        voiced[:min_len_loc],
    )
    prosody_np = np.stack(
        [f0 / 300.0, d_f0 / 100.0, rms / (rms.max() + 1e-8), voiced], axis=1
    )
    prosody_seq = torch.tensor(prosody_np, dtype=torch.float32, device=device)

    # --- 2) Feature extraction + projection to latent space ---
    if use_fast_features:
        # Lightweight path: MFCC -> SVD projection, aligned to prosody length.
        mfcc = librosa.feature.mfcc(
            y=y_in,
            sr=sr,
            n_mfcc=cfg["n_mfcc"],
            n_fft=frame_len_loc,
            hop_length=hop_len_loc,
            center=False,
        )
        feats = mfcc.T  # (T_mfcc, n_mfcc)

        # Align to prosody length
        T_target = prosody_seq.shape[0]
        if feats.shape[0] >= T_target:
            feats = feats[:T_target]
        else:
            pad_count = T_target - feats.shape[0]
            feats = np.vstack([feats, np.repeat(feats[-1:, :], pad_count, axis=0)])

        # SVD projection to state_dim
        Zc_h = feats - feats.mean(axis=0, keepdims=True)
        U_svd_h, Svals_h, Vt_h = np.linalg.svd(Zc_h, full_matrices=False)
        state_dim_local = cfg["state_dim"]
        Vtop_h = Vt_h[:state_dim_local, :]
        z_proj_np = Zc_h @ Vtop_h.T
        z_proj_latent = torch.tensor(z_proj_np, dtype=torch.float32, device=device)
    else:
        # Original HuBERT path: use existing global feat/model if available; otherwise load lazily.
        global feat, model
        try:
            _ = feat
            _ = model
        except NameError:
            from transformers import Wav2Vec2FeatureExtractor, HubertModel

            feat = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
            model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
            model.eval()

        inputs = feat(y_in, sampling_rate=sr, return_tensors="pt", padding="longest")
        with torch.no_grad():
            out = model(inputs["input_values"].to(device))
        hidden_local = out.last_hidden_state[0].cpu().numpy()  # (T_model, D)

        # Resample/interpolate HuBERT frames to prosody length
        T_model, feat_dim_local = hidden_local.shape
        t_model = np.linspace(0, 1, T_model)
        t_target = np.linspace(0, 1, prosody_seq.shape[0])
        hidden_resampled_local = np.stack(
            [np.interp(t_target, t_model, hidden_local[:, d]) for d in range(feat_dim_local)],
            axis=1,
        )

        H = torch.tensor(hidden_resampled_local, dtype=torch.float32, device=device)

        # Project HuBERT features to latent space: use probe_model if provided,
        # otherwise fall back to an SVD/PCA-style linear projection to state_dim.
        if probe_model is not None:
            with torch.no_grad():
                z_proj_latent = probe_model(H)  # (T, state_dim)
        else:
            hidden_np = hidden_resampled_local
            Zc_h = hidden_np - hidden_np.mean(axis=0, keepdims=True)
            U_svd_h, Svals_h, Vt_h = np.linalg.svd(Zc_h, full_matrices=False)
            state_dim_local = cfg["state_dim"]
            Vtop_h = Vt_h[:state_dim_local, :]
            z_proj_np = Zc_h @ Vtop_h.T
            z_proj_latent = torch.tensor(z_proj_np, dtype=torch.float32, device=device)

    # Deproject via learned U (Learned Deprojection U block)
    z_lat_seq = U_model(z_proj_latent)

    # --- 3) Adaptive Kalman filter over time to get latent state mu_t ---
    T_lat = z_lat_seq.shape[0]
    mu = torch.zeros(state_dim, device=device)
    P = torch.ones(state_dim, device=device) * cfg["kf_P0"]
    mus_list: list[torch.Tensor] = []
    for t in range(T_lat):
        out = kf_model.forward_step(mu, P, z_lat_seq[t], prosody_seq[t])
        mu, P, Q_t, R_t, K_t, gamma_t = out
        mus_list.append(mu.clone())
    latent_seq = torch.stack(mus_list, dim=0)  # (T, state_dim)

    # --- 4) Latent -> mel with baseline + PRN residual + smoothing ---
    base_decoder = LatentToMel(state_dim=latent_seq.shape[1], n_mels=n_mels).to(device)
    prn = ResidualMelPRN(
        state_dim=latent_seq.shape[1],
        prosody_dim=prosody_seq.shape[1],
        n_mels=n_mels,
        history_steps=history_steps,
    ).to(device)
    smoother = MelSmoother(n_mels=n_mels).to(device)

    # NOTE: In a full system these would be trained; here they are randomly
    # initialized and mainly serve to complete the architectural path.

    mel_norm = decode_mel_from_latent(
        latent_seq, prosody_seq, base_decoder, prn, smoother, history_steps
    )

    # For now we treat normalized mel as dB and "denormalize" as identity or
    # using cached stats if available.
    try:
        mel_db = denormalize_mel(mel_norm).detach().cpu().numpy()
    except AssertionError:
        mel_db = mel_norm.detach().cpu().numpy()

    # --- 5) Vocoder (Griffin-Lim) ---
    voc = GriffinLimVocoder(sr=sr, n_fft=1024, hop_length=hop_len_loc, n_mels=n_mels)
    waveform = voc(mel_db)

    return waveform, mel_db

