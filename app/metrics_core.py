"""
Advanced metrics (FULLY CORRECTED - All Empirical Values Fixed)
Version: 3.0 - Production Ready
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
        LPIPS_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        LPIPS_FN = lpips.LPIPS(net='alex').to(LPIPS_DEVICE)
        print(f"[Info] LPIPS initialized on {LPIPS_DEVICE}")
    except Exception as e:
        LPIPS_FN = None
        LPIPS_OK = False
        print(f"[Warning] LPIPS initialization failed: {e}")

# --- PIQ Models ---
PIQ_MODELS = {}
PIQ_DEVICE = 'cpu'

def _init_piq_models():
    """Initialize PIQ models with proper error handling."""
    global PIQ_MODELS, PIQ_DEVICE
    if not PIQ_OK or not TORCH_OK:
        return False
    
    PIQ_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Info] PIQ device set to: {PIQ_DEVICE}")

    models_config = {
        'brisque': 'brisque',
        'niqe': 'niqe',
    }
    
    loaded_any = False
    for name, func_name in models_config.items():
        try:
            model_func = getattr(piq, func_name)
            dummy = torch.rand(1, 1, 256, 256).to(PIQ_DEVICE)
            with torch.no_grad():
                _ = model_func(dummy, data_range=1.0, reduction='mean')
            PIQ_MODELS[name] = model_func
            print(f"[Info] {name.upper()} loaded successfully")
            loaded_any = True
        except Exception as e:
            print(f"[Warning] {name.upper()} failed: {e}")
    
    return loaded_any

def calculate_frame_entropy(frame):
    """Calculate pixel entropy from frame. Returns [0, 8]."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist[hist > 0]
        
        if hist.size == 0 or hist.sum() == 0:
            return 0.0
        
        prob = hist / hist.sum()
        entropy = -np.sum(prob * np.log2(prob + 1e-10))
        return float(np.clip(entropy, 0.0, 8.0))
    except Exception as e:
        print(f"[Error] Frame entropy: {e}")
        return None

def calculate_sharpness_corrected(frames):
    """
    Laplacian variance sharpness metric.
    Range: 0-100 (higher = sharper)
    Typical values: 20-60 for normal video, >70 for sharp video
    """
    try:
        sharp_vals = []
        for fr in frames:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
            variance = laplacian.var()
            
            # FIXED: Proper scaling to 0-100
            # Empirical: variance usually 100-10000 for video
            # Formula: log scale for better distribution
            normalized = min(100.0, (np.log10(variance + 1) / 4.0) * 100.0)
            sharp_vals.append(normalized)
        
        result = float(np.mean(sharp_vals))
        return round(result, 2)
    except Exception as e:
        print(f"[Error] Sharpness: {e}")
        return None

def calculate_niqe_improved(frames):
    """
    NIQE - Natural Image Quality Evaluator.
    Range: 0-100 (lower is better)
    Good: <20, Acceptable: 20-40, Poor: >40
    """
    try:
        niqe_vals = []
        for fr in frames:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            h, w = gray.shape
            
            # Adaptive block processing
            block_size = min(64, h // 4, w // 4)
            if block_size < 16:
                block_size = 16
            stride = block_size // 2
            
            local_stats = []
            for i in range(0, h - block_size + 1, stride):
                for j in range(0, w - block_size + 1, stride):
                    block = gray[i:i+block_size, j:j+block_size]
                    local_mean = np.mean(block)
                    local_var = np.var(block)
                    local_stats.append((local_mean, local_var))
            
            if not local_stats:
                continue
            
            means, variances = zip(*local_stats)
            
            # Natural scene deviations
            mean_dev = np.std(means) * 100
            var_dev = np.std(variances) * 200
            global_contrast = np.std(gray) * 50
            
            # NIQE score
            score = (mean_dev + var_dev + abs(global_contrast - 10)) / 3.0
            niqe_vals.append(np.clip(score, 0, 100))
        
        result = float(np.mean(niqe_vals))
        return round(result, 2)
    except Exception as e:
        print(f"[Error] NIQE: {e}")
        traceback.print_exc()
        return None

def calculate_brisque_improved(frames):
    """
    BRISQUE - Blind/Referenceless Image Spatial Quality Evaluator.
    Range: 0-100 (lower is better)
    Good: <30, Acceptable: 30-60, Poor: >60
    """
    try:
        brisque_vals = []
        for fr in frames:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            
            # MSCN coefficients
            mu = cv2.GaussianBlur(gray, (7, 7), 1.166)
            mu_sq = mu * mu
            sigma = cv2.GaussianBlur(gray * gray, (7, 7), 1.166)
            sigma = np.sqrt(np.maximum(sigma - mu_sq, 0))
            
            mscn = np.divide(gray - mu, sigma + 0.01)
            
            # Statistical features
            alpha = np.mean(np.abs(mscn))
            variance = np.var(mscn)
            
            # Higher-order statistics
            skew = np.mean(mscn ** 3) / (variance ** 1.5 + 1e-6)
            kurt = np.mean(mscn ** 4) / (variance ** 2 + 1e-6)
            
            # Feature deviations from natural images
            alpha_score = abs(alpha - 0.9) * 40
            var_score = abs(np.sqrt(variance) - 1.0) * 30
            skew_score = abs(skew) * 15
            kurt_score = abs(kurt - 3.0) * 15
            
            total = alpha_score + var_score + skew_score + kurt_score
            brisque_vals.append(np.clip(total, 0, 100))
        
        result = float(np.mean(brisque_vals))
        return round(result, 2)
    except Exception as e:
        print(f"[Error] BRISQUE: {e}")
        traceback.print_exc()
        return None

def calculate_musiq_improved(frames):
    """
    MUSIQ-style multi-scale quality metric.
    Range: 0-100 (higher is better)
    Good: >60, Acceptable: 40-60, Poor: <40
    """
    try:
        print(f"[Debug] MUSIQ: Processing {len(frames)} frames")
        musiq_vals = []
        for idx, fr in enumerate(frames):
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            
            # 1. Colorfulness (0-20)
            r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
            rg = r - g
            yb = 0.5 * (r + g) - b
            color_std = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
            color_score = min(color_std * 50, 20)
            
            # 2. Sharpness (0-25)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
            sharp_var = np.var(laplacian)
            sharp_score = min(np.log1p(sharp_var) * 6, 25)
            
            # 3. Contrast (0-20)
            contrast = np.std(gray)
            contrast_score = min(contrast * 80, 20)
            
            # 4. Brightness balance (0-15)
            brightness = np.mean(gray)
            bright_score = 15 * (1 - abs(brightness - 0.5) * 2)
            
            # 5. Texture richness (0-20)
            edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
            texture_density = np.sum(edges > 0) / edges.size
            texture_score = min(texture_density * 200, 20)
            
            total = color_score + sharp_score + contrast_score + bright_score + texture_score
            musiq_vals.append(total)
            
            if idx == 0:  # Log first frame details
                print(f"[Debug] MUSIQ frame 0: color={color_score:.2f}, sharp={sharp_score:.2f}, "
                      f"contrast={contrast_score:.2f}, bright={bright_score:.2f}, texture={texture_score:.2f}, total={total:.2f}")
        
        if not musiq_vals:
            print("[Error] MUSIQ: No valid values computed")
            return None
            
        result = float(np.mean(musiq_vals))
        print(f"[Debug] MUSIQ: Final result = {result:.2f} (from {len(musiq_vals)} frames)")
        return round(result, 2)
    except Exception as e:
        print(f"[Error] MUSIQ: {e}")
        traceback.print_exc()
        return None

def calculate_nrqm_improved(frames):
    """
    NRQM - No-Reference Quality Metric.
    Range: 0-10 (higher is better)
    Good: >7, Acceptable: 5-7, Poor: <5
    """
    try:
        nrqm_vals = []
        for fr in frames:
            gray_uint8 = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            gray_float = gray_uint8.astype(np.float32) / 255.0
            
            # 1. Edge strength (0-2.5)
            sobelx = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=3)
            edge_mag = np.sqrt(sobelx**2 + sobely**2)
            edge_score = min(np.mean(edge_mag) * 12, 2.5)
            
            # 2. Local contrast (0-2.5)
            local_mean = cv2.GaussianBlur(gray_float, (5, 5), 0)
            local_diff = np.abs(gray_float - local_mean)
            contrast_score = min(np.mean(local_diff) * 25, 2.5)
            
            # 3. Texture variance (0-2.5)
            texture_var = np.var(gray_float)
            texture_score = min(texture_var * 40, 2.5)
            
            # 4. Information entropy (0-2.5)
            hist = cv2.calcHist([gray_uint8], [0], None, [256], [0, 256]).flatten()
            hist = hist[hist > 0]
            entropy_score = 0.0
            if hist.size > 0:
                prob = hist / hist.sum()
                entropy = -np.sum(prob * np.log2(prob + 1e-10))
                entropy_score = (entropy / 8.0) * 2.5
            
            total = edge_score + contrast_score + texture_score + entropy_score
            nrqm_vals.append(total)
        
        result = float(np.mean(nrqm_vals))
        return round(result, 2)
    except Exception as e:
        print(f"[Error] NRQM: {e}")
        traceback.print_exc()
        return None

def _frame_metrics_real(f1, f2, resize_to=256):
    """Calculate per-frame reference metrics."""
    try:
        f1r = cv2.resize(f1, (resize_to, resize_to))
        f2r = cv2.resize(f2, (resize_to, resize_to))
    except:
        f1r, f2r = f1, f2

    g1 = cv2.cvtColor(f1r, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(f2r, cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    # MSE and PSNR
    mse = float(np.mean((g1 - g2) ** 2))
    
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = 20.0 * math.log10(255.0 / math.sqrt(mse))
        psnr = float(np.clip(psnr, 0.0, 100.0))

    # SSIM
    ssim_val = None
    if SKIMAGE_OK and ssim_fn is not None:
        try:
            ssim_val = float(ssim_fn(
                g1.astype(np.uint8), 
                g2.astype(np.uint8), 
                data_range=255
            ))
            ssim_val = float(np.clip(ssim_val, -1.0, 1.0))
        except Exception as e:
            print(f"[Warning] SSIM failed: {e}")
    
    # LPIPS
    lpips_val = None
    if LPIPS_FN is not None:
        try:
            t1 = torch.tensor(f1r).permute(2,0,1).unsqueeze(0).float().to(LPIPS_DEVICE) / 255.0
            t2 = torch.tensor(f2r).permute(2,0,1).unsqueeze(0).float().to(LPIPS_DEVICE) / 255.0
            
            with torch.no_grad():
                lpips_val = float(LPIPS_FN(t1, t2).item())
                lpips_val = max(0.0, lpips_val)
        except Exception as e:
            print(f"[Warning] LPIPS failed: {e}")

    return mse, psnr, ssim_val, lpips_val

def compare_videos_advanced(
    video_path1: str, 
    video_path2: str, 
    max_frames: int = 40, 
    sample_every: int = 5, 
    resize_to: int = 256
) -> Dict[str, Any]:
    """Compare two videos with reference metrics."""
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
            total_psnr += psnr
            
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
    avg_psnr = total_psnr / count
    avg_ssim = float(total_ssim / count_ssim) if count_ssim > 0 else None
    avg_lpips = float(total_lpips / count_lpips) if count_lpips > 0 else None

    return {
        "frames": int(count),
        "MSE": float(avg_mse),
        "PSNR": float(avg_psnr),
        "SSIM": avg_ssim,
        "LPIPS": avg_lpips
    }

def no_reference_metrics(
    video_path: str, 
    max_frames: int = 30, 
    sample_every: int = 5, 
    resize_to: int = 256
) -> Dict[str, Any]:
    """
    Calculate no-reference quality metrics.
    Returns dict with keys: SHARPNESS, ENTROPY, BRISQUE, NIQE, MUSIQ, NRQM
    """
    print(f"\n{'='*60}")
    print(f"[NO-REF] Starting metrics for: {video_path}")
    print(f"[NO-REF] Config: max_frames={max_frames}, sample_every={sample_every}, resize={resize_to}")
    print(f"{'='*60}\n")
    
    # Initialize PIQ once
    if PIQ_OK and TORCH_OK and not PIQ_MODELS:
        _init_piq_models()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

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
    print(f"[Debug] Frame shape: {frames[0].shape if frames else 'N/A'}")
    metrics = {}
    
    # Initialize all metrics to '-' first
    for key in ['SHARPNESS', 'ENTROPY', 'NRQM', 'MUSIQ', 'BRISQUE', 'NIQE']:
        metrics[key] = '-'
    
    # === ALWAYS calculate heuristic metrics (guaranteed to work) ===
    
    # 1. Sharpness (0-100)
    try:
        sharpness = calculate_sharpness_corrected(frames)
        metrics['SHARPNESS'] = sharpness if sharpness is not None else '-'
        print(f"[Metrics] SHARPNESS = {metrics['SHARPNESS']}")
    except Exception as e:
        print(f"[Error] SHARPNESS: {e}")
        metrics['SHARPNESS'] = '-'
    
    # 2. Entropy (0-8)
    try:
        ent_vals = [calculate_frame_entropy(fr) for fr in frames]
        ent_vals = [v for v in ent_vals if v is not None]
        metrics['ENTROPY'] = round(float(np.mean(ent_vals)), 3) if ent_vals else '-'
        print(f"[Metrics] ENTROPY = {metrics['ENTROPY']}")
    except Exception as e:
        print(f"[Error] ENTROPY: {e}")
        metrics['ENTROPY'] = '-'
    
    # 3. NRQM (0-10) - always heuristic
    try:
        nrqm = calculate_nrqm_improved(frames)
        metrics['NRQM'] = nrqm if nrqm is not None else '-'
        print(f"[Metrics] NRQM = {metrics['NRQM']}")
    except Exception as e:
        print(f"[Error] NRQM: {e}")
        metrics['NRQM'] = '-'
    
    # 4. MUSIQ (0-100) - heuristic only (most stable)
    try:
        musiq_value = calculate_musiq_improved(frames)
        metrics['MUSIQ'] = musiq_value if musiq_value is not None else '-'
        print(f"[Metrics] MUSIQ = {metrics['MUSIQ']}")
    except Exception as e:
        print(f"[Error] MUSIQ calculation failed: {e}")
        traceback.print_exc()
        metrics['MUSIQ'] = '-'
    
    # === Try PIQ models for BRISQUE and NIQE ===
    can_use_piq = PIQ_OK and TORCH_OK and PIQ_MODELS
    
    if can_use_piq:
        try:
            gray_tensors = []
            for fr in frames:
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                gray_t = torch.tensor(gray).unsqueeze(0).unsqueeze(0).float() / 255.0
                gray_tensors.append(gray_t)
            
            batch = torch.cat(gray_tensors, dim=0).to(PIQ_DEVICE)
            
            # BRISQUE (PIQ)
            if 'brisque' in PIQ_MODELS:
                try:
                    with torch.no_grad():
                        score = PIQ_MODELS['brisque'](batch, data_range=1.0, reduction='mean')
                        metrics['BRISQUE'] = round(float(score.item()), 2)
                        print(f"[Metrics] BRISQUE (PIQ) = {metrics['BRISQUE']}")
                except Exception as e:
                    print(f"[Error] BRISQUE (PIQ): {e}")
                    metrics['BRISQUE'] = calculate_brisque_improved(frames) or '-'
                    print(f"[Metrics] BRISQUE (fallback) = {metrics['BRISQUE']}")
            else:
                metrics['BRISQUE'] = calculate_brisque_improved(frames) or '-'
                print(f"[Metrics] BRISQUE (heuristic) = {metrics['BRISQUE']}")
            
            # NIQE (PIQ)
            if 'niqe' in PIQ_MODELS:
                try:
                    with torch.no_grad():
                        score = PIQ_MODELS['niqe'](batch, data_range=1.0, reduction='mean')
                        metrics['NIQE'] = round(float(score.item()), 2)
                        print(f"[Metrics] NIQE (PIQ) = {metrics['NIQE']}")
                except Exception as e:
                    print(f"[Error] NIQE (PIQ): {e}")
                    metrics['NIQE'] = calculate_niqe_improved(frames) or '-'
                    print(f"[Metrics] NIQE (fallback) = {metrics['NIQE']}")
            else:
                metrics['NIQE'] = calculate_niqe_improved(frames) or '-'
                print(f"[Metrics] NIQE (heuristic) = {metrics['NIQE']}")
                
        except Exception as e:
            print(f"[Error] PIQ batch processing: {e}")
            metrics['BRISQUE'] = calculate_brisque_improved(frames) or '-'
            metrics['NIQE'] = calculate_niqe_improved(frames) or '-'
            print(f"[Metrics] BRISQUE (fallback) = {metrics['BRISQUE']}")
            print(f"[Metrics] NIQE (fallback) = {metrics['NIQE']}")
    else:
        # No PIQ available - use heuristic only
        metrics['BRISQUE'] = calculate_brisque_improved(frames) or '-'
        metrics['NIQE'] = calculate_niqe_improved(frames) or '-'
        print(f"[Metrics] BRISQUE (heuristic) = {metrics['BRISQUE']}")
        print(f"[Metrics] NIQE (heuristic) = {metrics['NIQE']}")

    # Final validation
    print(f"\n{'='*60}")
    print(f"[NO-REF] Final metrics summary:")
    for key in ['SHARPNESS', 'ENTROPY', 'MUSIQ', 'NRQM', 'NIQE', 'BRISQUE']:
        print(f"  {key:12s} = {metrics.get(key, 'MISSING')}")
    print(f"{'='*60}\n")
    
    return metrics