import math
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_metric

# ---------- File-level ----------
def file_entropy(path: str) -> float:
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size == 0:
        return 0.0
    counts = np.bincount(data, minlength=256).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def chi_square_bytes(path: str):
    """(chi2, dof, p_approx) untuk uniformitas 256 bin."""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    n = data.size
    if n == 0:
        return 0.0, 255, 1.0
    obs = np.bincount(data, minlength=256).astype(np.float64)
    exp = n / 256.0
    chi2 = np.sum((obs - exp) ** 2 / exp)
    dof = 255
    # Wilson–Hilferty (tanpa scipy)
    c = (chi2 / dof) ** (1/3)
    z = (c - (1 - 2/(9*dof))) / math.sqrt(2/(9*dof))
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return float(chi2), dof, float(p)

# ---------- Video helpers ----------
def _open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap

def _iter_frames_synchronized(pathA, pathB, step=15, max_frames=150):
    capA, capB = _open_video(pathA), _open_video(pathB)
    try:
        idx, yielded = 0, 0
        while True:
            capA.set(cv2.CAP_PROP_POS_FRAMES, idx)
            capB.set(cv2.CAP_PROP_POS_FRAMES, idx)
            okA, fa = capA.read()
            okB, fb = capB.read()
            if not (okA and okB):
                break
            h = min(fa.shape[0], fb.shape[0])
            w = min(fa.shape[1], fb.shape[1])
            fa = cv2.resize(fa, (w, h), interpolation=cv2.INTER_AREA)
            fb = cv2.resize(fb, (w, h), interpolation=cv2.INTER_AREA)
            ya = cv2.cvtColor(fa, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            yb = cv2.cvtColor(fb, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            yield ya, yb
            yielded += 1
            if yielded >= max_frames:
                break
            idx += step
    finally:
        capA.release(); capB.release()

def psnr_ssim_between_videos(path_ref, path_cmp, step=15, max_frames=150):
    psnrs, ssims = [], []
    for ya, yb in _iter_frames_synchronized(path_ref, path_cmp, step, max_frames):
        mse = np.mean((ya.astype(np.float64) - yb.astype(np.float64)) ** 2)
        psnrs.append(100.0 if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse))
        ssims.append(float(ssim_metric(ya, yb, data_range=255)))
    return float(np.mean(psnrs)) if psnrs else 0.0, float(np.mean(ssims)) if ssims else 0.0

def sharpness_video(path, step=15, max_frames=150):
    vals = []
    cap = _open_video(path)
    try:
        idx, yielded = 0, 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, f = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            yielded += 1
            if yielded >= max_frames:
                break
            idx += step
    finally:
        cap.release()
    return float(np.mean(vals)) if vals else 0.0

# ---------- Wrappers untuk tabel ----------
def metrics_for_original(orig_path):
    ent = file_entropy(orig_path)
    chi2, dof, p = chi_square_bytes(orig_path)
    sharp = sharpness_video(orig_path)
    return {
        "Jenis": "Asli",
        "PSNR": "-", "SSIM": "-", "LPIPS": "-",
        "MUSIQ": "-", "NRQM": "-", "NIQE": "-",
        "BRISQUE": "-", "Sharpness": sharp,
        "FileEntropy": ent, "ChiSquare": chi2, "ChiDoF": dof, "ChiP": p
    }

def metrics_for_encrypted(enc_path):
    ent = file_entropy(enc_path)
    chi2, dof, p = chi_square_bytes(enc_path)
    return {
        "Jenis": "Terenkripsi",
        "PSNR": "N/A", "SSIM": "N/A", "LPIPS": "N/A",
        "MUSIQ": "N/A", "NRQM": "N/A", "NIQE": "N/A",
        "BRISQUE": "N/A", "Sharpness": "N/A",
        "FileEntropy": ent, "ChiSquare": chi2, "ChiDoF": dof, "ChiP": p
    }

def metrics_for_decrypted(orig_path, dec_path):
    psnr, ssim = psnr_ssim_between_videos(orig_path, dec_path)
    ent = file_entropy(dec_path)
    chi2, dof, p = chi_square_bytes(dec_path)
    sharp = sharpness_video(dec_path)
    return {
        "Jenis": "Didekripsi",
        "PSNR": psnr, "SSIM": ssim, "LPIPS": 0.0,
        "MUSIQ": "-", "NRQM": "-", "NIQE": "-",
        "BRISQUE": "-", "Sharpness": sharp,
        "FileEntropy": ent, "ChiSquare": chi2, "ChiDoF": dof, "ChiP": p
    }
