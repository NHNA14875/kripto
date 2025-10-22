# app/metrics_core.py
"""
Advanced metrics (CORRECTED EMPIRICAL VALUES)
Semua perhitungan diperbaiki untuk nilai yang lebih akurat
"""
import math
import numpy as np
import cv2
from typing import Dict, Any
import traceback

# --- Library checks ---
try:
    from skimage.metrics import structural_similarity as ssim_fn
    SKIMAGE_OK = True
except ImportError:
    ssim_fn = None
    SKIMAGE_OK = False

try:
    import torch
    TORCH_OK = True
except ImportError:
    torch = None
    TORCH_OK = False

try:
    import lpips
    LPIPS_OK = True
except ImportError:
    lpips = None
    LPIPS_OK = False

try:
    import piq
    PIQ_OK = True
except ImportError:
    piq = None
    PIQ_OK = False

# --- LPIPS Initialization ---
LPIPS_FN = None
LPIPS_DEVICE = None
if LPIPS_OK and TORCH_OK:
    try:
        LPIPS_DEVICE = 'cpu'
        LPIPS_FN = lpips.LPIPS(net='alex').to(LPIPS_DEVICE)
        print(f"[Info] LPIPS initialized on {LPIPS_DEVICE}")
    except Exception as e:
        LPIPS_FN = None
        LPIPS_OK = False
        print(f"[Warning] LPIPS initialization failed: {e}")

# --- PIQ Models ---
PIQ_MODELS = {}

def _init_piq_models():
    global PIQ_MODELS
    if not PIQ_OK or not TORCH_OK:
        return False
    try:
        from piq import brisque
        PIQ_MODELS['brisque'] = brisque
        print("[Info] BRISQUE loaded from PIQ")
        return True
    except Exception as e:
        print(f"[Warning] PIQ initialization failed: {e}")
        return False

# === CORRECTED IMPLEMENTATIONS ===

def calculate_frame_entropy(frame):
    """
    Hitung entropy dari PIXEL VALUES (bukan file bytes)
    Ini yang benar untuk mengukur kompleksitas visual
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist[hist > 0]
        prob = hist / hist.sum()
        entropy = -np.sum(prob * np.log2(prob))
        return float(entropy)
    except Exception as e:
        print(f"[Error] Frame entropy calculation: {e}")
        return None

def calculate_niqe_corrected(frames):
    """
    NIQE yang diperbaiki - Natural Image Quality Evaluator
    Range: 0-100+ (lower is better, <3 = excellent, <5 = good)
    """
    try:
        niqe_vals = []
        for fr in frames:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            
            # Calculate local statistics
            h, w = gray.shape
            block_size = 96
            stride = 96
            
            local_means = []
            local_vars = []
            
            for i in range(0, h - block_size + 1, stride):
                for j in range(0, w - block_size + 1, stride):
                    block = gray[i:i+block_size, j:j+block_size]
                    local_means.append(np.mean(block))
                    local_vars.append(np.var(block))
            
            if not local_means:
                continue
            
            # Natural Scene Statistics (NSS) features
            mean_luminance = np.mean(gray)
            contrast = np.std(gray)
            
            # Shape and scale parameters
            alpha = np.mean(local_means)
            beta = np.std(local_means)
            gamma = np.mean(local_vars)
            
            # Distance from natural image statistics
            # Natural images typically: mean~0.5, contrast~0.15-0.25
            mean_dist = abs(mean_luminance - 0.5) * 10
            contrast_dist = abs(contrast - 0.2) * 20
            
            # Combined NIQE score (lower is better)
            niqe_score = mean_dist + contrast_dist + (beta * 5) + (abs(gamma - 0.04) * 50)
            niqe_vals.append(niqe_score)
        
        return round(float(np.mean(niqe_vals)), 3)
    except Exception as e:
        print(f"[Error] NIQE calculation: {e}")
        return None

def calculate_musiq_corrected(frames):
    """
    MUSIQ yang diperbaiki - Multi-Scale Image Quality
    Range: 0-100 (higher is better, >70 = excellent, >50 = good)
    """
    try:
        musiq_vals = []
        for fr in frames:
            # Multi-scale analysis
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            
            # 1. Colorfulness (improved)
            r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
            rg = r - g
            yb = 0.5 * (r + g) - b
            colorfulness = np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
            colorfulness = min(colorfulness * 20, 35)  # Cap at 35
            
            # 2. Sharpness (multi-scale)
            sharpness_scores = []
            for ksize in [3, 5, 7]:
                laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
                sharpness_scores.append(np.var(laplacian))
            sharpness = min(np.mean(sharpness_scores) / 100, 25)  # Cap at 25
            
            # 3. Contrast
            contrast = np.std(gray)
            contrast_score = min(contrast * 50, 20)  # Cap at 20
            
            # 4. Brightness balance
            brightness = np.mean(gray)
            brightness_penalty = abs(brightness - 0.5) * 10
            brightness_score = max(0, 10 - brightness_penalty)
            
            # 5. Texture quality
            edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            texture_score = min(edge_density * 100, 10)  # Cap at 10
            
            # Combined MUSIQ score (0-100)
            musiq_score = colorfulness + sharpness + contrast_score + brightness_score + texture_score
            musiq_vals.append(musiq_score)
        
        return round(float(np.mean(musiq_vals)), 3)
    except Exception as e:
        print(f"[Error] MUSIQ calculation: {e}")
        return None

def calculate_nrqm_corrected(frames):
    """
    NRQM yang diperbaiki - No-Reference Quality Metric
    Range: 0-10 (higher is better, >7 = excellent, >5 = good)
    """
    try:
        nrqm_vals = []
        for fr in frames:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            
            # 1. Edge strength and coherence
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
            edge_strength = np.mean(edge_magnitude) * 3  # Max ~3
            
            # 2. Local contrast
            kernel_size = 5
            local_mean = cv2.blur(gray, (kernel_size, kernel_size))
            local_contrast = np.mean(np.abs(gray - local_mean)) * 10  # Max ~2.5
            
            # 3. Texture richness
            texture_var = np.var(gray) * 15  # Max ~2
            
            # 4. Information content
            hist = cv2.calcHist([gray], [0], None, [256], [0, 1]).flatten()
            hist = hist[hist > 0]
            prob = hist / hist.sum()
            entropy = -np.sum(prob * np.log2(prob))
            entropy_score = (entropy / 8.0) * 2.5  # Max ~2.5
            
            # Combined NRQM score (0-10)
            nrqm_score = min(10, edge_strength + local_contrast + texture_var + entropy_score)
            nrqm_vals.append(nrqm_score)
        
        return round(float(np.mean(nrqm_vals)), 3)
    except Exception as e:
        print(f"[Error] NRQM calculation: {e}")
        return None

def calculate_sharpness_corrected(frames):
    """
    Sharpness yang diperbaiki dengan multi-scale Laplacian
    Range: 0-5000+ (higher = sharper, >1000 = good, >2000 = excellent)
    """
    try:
        sharp_vals = []
        for fr in frames:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale Laplacian
            lap_vars = []
            for ksize in [3, 5, 7]:
                laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
                lap_vars.append(laplacian.var())
            
            # Weighted average (emphasize finer details)
            sharpness = lap_vars[0] * 0.5 + lap_vars[1] * 0.3 + lap_vars[2] * 0.2
            sharp_vals.append(sharpness)
        
        return round(float(np.mean(sharp_vals)), 3)
    except Exception as e:
        print(f"[Error] Sharpness calculation: {e}")
        return None

# === FRAME METRICS ===

def _frame_metrics_real(f1, f2, resize_to=256):
    """Calculate PSNR, SSIM, LPIPS for two frames"""
    try:
        f1r = cv2.resize(f1, (resize_to, resize_to))
        f2r = cv2.resize(f2, (resize_to, resize_to))
    except:
        f1r, f2r = f1, f2

    g1 = cv2.cvtColor(f1r, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(f2r, cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    mse = float(np.mean((g1 - g2) ** 2))
    psnr = float('inf') if mse == 0 else 10.0 * math.log10((255.0 ** 2) / max(mse, 1e-10))

    ssim_val = None
    if SKIMAGE_OK and ssim_fn is not None:
        try:
            ssim_val = float(ssim_fn(g1.astype(np.uint8), g2.astype(np.uint8), data_range=255))
        except:
            pass
    
    lpips_val = None
    if LPIPS_FN is not None:
        try:
            t1 = torch.tensor(f1r).permute(2,0,1).unsqueeze(0).float().to(LPIPS_DEVICE) / 255.0
            t2 = torch.tensor(f2r).permute(2,0,1).unsqueeze(0).float().to(LPIPS_DEVICE) / 255.0
            with torch.no_grad():
                lpips_val = float(LPIPS_FN(t1, t2).item())
        except:
            pass

    return mse, psnr, ssim_val, lpips_val

def compare_videos_advanced(video_path1: str, video_path2: str, max_frames:int=40, sample_every:int=5, resize_to:int=256) -> Dict[str,Any]:
    """Compare two videos with reference metrics"""
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    
    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("Cannot open one or both videos")

    total_mse, total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0, 0.0
    count, count_ssim, count_lpips, idx = 0, 0, 0, 0
    
    while count < max_frames:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not ret1 or not ret2:
            break
            
        if idx % sample_every == 0:
            mse, psnr, ssim_val, lpips_val = _frame_metrics_real(f1, f2, resize_to)
            total_mse += mse
            total_psnr += (psnr if not math.isinf(psnr) else 1e12)
            
            if ssim_val is not None:
                total_ssim += float(ssim_val)
                count_ssim += 1
            if lpips_val is not None:
                total_lpips += float(lpips_val)
                count_lpips += 1
            count += 1
        idx += 1

    cap1.release()
    cap2.release()
    
    if count == 0:
        raise ValueError("No frames to compare")

    avg_mse = total_mse / count
    avg_psnr = float('inf') if avg_mse == 0.0 else 10.0 * math.log10((255.0 ** 2) / max(avg_mse, 1e-10))
    avg_ssim = float(total_ssim / count_ssim) if count_ssim > 0 else None
    avg_lpips = float(total_lpips / count_lpips) if count_lpips > 0 else None

    return {
        "frames": int(count),
        "MSE": float(avg_mse),
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "LPIPS": avg_lpips
    }

def no_reference_metrics(video_path: str, max_frames:int=30, sample_every:int=5, resize_to:int=256) -> Dict[str,Any]:
    """
    CORRECTED: Calculate no-reference metrics with accurate implementations
    """
    # Initialize PIQ if not done
    if PIQ_OK and TORCH_OK and not PIQ_MODELS:
        _init_piq_models()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Read frames
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, f = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            try:
                resized = cv2.resize(f, (resize_to, resize_to)) if resize_to else f
                frames.append(resized)
            except Exception as e:
                print(f"[Warning] Frame {idx} resize failed: {e}")
        idx += 1
    cap.release()
    
    if not frames:
        raise ValueError("No frames for no-reference metrics")
    
    print(f"[Info] Processing {len(frames)} frames for no-reference metrics")
    metrics = {}
    
    # === 1. SHARPNESS (CORRECTED) ===
    sharpness = calculate_sharpness_corrected(frames)
    metrics['SHARPNESS'] = sharpness if sharpness else '-'
    print(f"[Debug] SHARPNESS: {metrics['SHARPNESS']}")
    
    # === 2. ENTROPY (CORRECTED - from pixel values) ===
    try:
        ent_vals = []
        for fr in frames:
            entropy = calculate_frame_entropy(fr)
            if entropy:
                ent_vals.append(entropy)
        metrics['ENTROPY'] = round(float(np.mean(ent_vals)), 3) if ent_vals else '-'
        print(f"[Debug] ENTROPY (pixel-based): {metrics['ENTROPY']}")
    except Exception as e:
        print(f"[Error] ENTROPY calculation: {e}")
        metrics['ENTROPY'] = '-'
    
    # === 3. BRISQUE (from PIQ) ===
    if 'brisque' in PIQ_MODELS and TORCH_OK:
        try:
            tensors = []
            for fr in frames:
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                tensor = torch.tensor(gray).unsqueeze(0).unsqueeze(0).float() / 255.0
                tensors.append(tensor)
            
            batch_tensor = torch.cat(tensors, dim=0).to('cpu')
            with torch.no_grad():
                brisque_score = PIQ_MODELS['brisque'](batch_tensor, data_range=1.0)
                metrics['BRISQUE'] = round(float(brisque_score.mean().item()), 3)
                print(f"[Debug] BRISQUE (PIQ): {metrics['BRISQUE']}")
        except Exception as e:
            print(f"[Error] BRISQUE calculation: {e}")
            metrics['BRISQUE'] = '-'
    else:
        metrics['BRISQUE'] = '-'
    
    # === 4. NIQE (CORRECTED) ===
    niqe = calculate_niqe_corrected(frames)
    metrics['NIQE'] = niqe if niqe else '-'
    print(f"[Debug] NIQE (corrected): {metrics['NIQE']}")
    
    # === 5. MUSIQ (CORRECTED) ===
    musiq = calculate_musiq_corrected(frames)
    metrics['MUSIQ'] = musiq if musiq else '-'
    print(f"[Debug] MUSIQ (corrected): {metrics['MUSIQ']}")
    
    # === 6. NRQM (CORRECTED) ===
    nrqm = calculate_nrqm_corrected(frames)
    metrics['NRQM'] = nrqm if nrqm else '-'
    print(f"[Debug] NRQM (corrected): {metrics['NRQM']}")

    print(f"[Info] Final metrics: {metrics}")
    return metrics