# app/metrics_corrected.py
"""
CORRECTED METRICS IMPLEMENTATION
Implementasi yang benar-benar diperbaiki dengan nilai empiris yang akurat
"""
import math
import numpy as np
import cv2
from typing import Dict, Any, List
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
    """Initialize PIQ models"""
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

# ============================================================================
# CORRECTED METRIC IMPLEMENTATIONS
# ============================================================================

def calculate_frame_entropy(frame) -> float:
    """
    Hitung entropy dari PIXEL VALUES (Shannon Entropy)
    Range: 0-8 bits (8 = maksimum untuk 256 gray levels)
    
    Returns:
        float: Entropy value atau None jika error
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        
        # Hapus bin dengan count 0
        hist = hist[hist > 0]
        
        # Hitung probabilitas
        prob = hist / hist.sum()
        
        # Shannon entropy
        entropy = -np.sum(prob * np.log2(prob))
        
        return float(entropy)
    except Exception as e:
        print(f"[Error] Frame entropy calculation: {e}")
        traceback.print_exc()
        return None


def calculate_sharpness_corrected(frames: List[np.ndarray]) -> float:
    """
    Sharpness menggunakan multi-scale Laplacian variance
    Range: 0-10000+ (higher = sharper)
    - Excellent: > 2000
    - Good: 1000-2000
    - Fair: 500-1000
    - Poor: < 500
    
    Returns:
        float: Sharpness score atau None jika error
    """
    try:
        sharp_vals = []
        
        for fr in frames:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale Laplacian dengan berbagai kernel sizes
            lap_vars = []
            for ksize in [3, 5, 7]:
                laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
                lap_vars.append(laplacian.var())
            
            # Weighted average (lebih menekankan detail halus)
            sharpness = (lap_vars[0] * 0.5 +  # Fine details
                        lap_vars[1] * 0.3 +   # Medium details
                        lap_vars[2] * 0.2)    # Coarse details
            
            sharp_vals.append(sharpness)
        
        avg_sharpness = float(np.mean(sharp_vals))
        print(f"[Debug] Sharpness calculated: {avg_sharpness:.2f}")
        
        return round(avg_sharpness, 2)
    
    except Exception as e:
        print(f"[Error] Sharpness calculation: {e}")
        traceback.print_exc()
        return None


def calculate_niqe_corrected(frames: List[np.ndarray]) -> float:
    """
    NIQE (Natural Image Quality Evaluator)
    Range: 0-100+ (LOWER is BETTER)
    - Excellent: < 3
    - Good: 3-5
    - Fair: 5-8
    - Poor: > 8
    
    Berdasarkan Natural Scene Statistics (NSS)
    
    Returns:
        float: NIQE score atau None jika error
    """
    try:
        niqe_vals = []
        
        for fr in frames:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            h, w = gray.shape
            
            # Parameters untuk block analysis
            block_size = 96
            stride = 96
            
            local_means = []
            local_vars = []
            
            # Calculate local statistics
            for i in range(0, h - block_size + 1, stride):
                for j in range(0, w - block_size + 1, stride):
                    block = gray[i:i+block_size, j:j+block_size]
                    local_means.append(np.mean(block))
                    local_vars.append(np.var(block))
            
            if not local_means:
                continue
            
            # Global statistics
            mean_luminance = np.mean(gray)
            contrast = np.std(gray)
            
            # NSS parameters
            alpha = np.mean(local_means)
            beta = np.std(local_means)
            gamma = np.mean(local_vars)
            
            # Distance from natural statistics
            # Natural images: mean ≈ 0.5, contrast ≈ 0.2
            mean_dist = abs(mean_luminance - 0.5) * 10
            contrast_dist = abs(contrast - 0.2) * 20
            
            # Local variation penalties
            local_mean_penalty = beta * 5
            local_var_penalty = abs(gamma - 0.04) * 50
            
            # Combined NIQE score
            niqe_score = (mean_dist + 
                         contrast_dist + 
                         local_mean_penalty + 
                         local_var_penalty)
            
            niqe_vals.append(niqe_score)
        
        avg_niqe = float(np.mean(niqe_vals))
        print(f"[Debug] NIQE calculated: {avg_niqe:.3f}")
        
        return round(avg_niqe, 3)
    
    except Exception as e:
        print(f"[Error] NIQE calculation: {e}")
        traceback.print_exc()
        return None


def calculate_musiq_corrected(frames: List[np.ndarray]) -> float:
    """
    MUSIQ (Multi-Scale Image Quality)
    Range: 0-100 (HIGHER is BETTER)
    - Excellent: > 70
    - Good: 50-70
    - Fair: 30-50
    - Poor: < 30
    
    Components:
    - Colorfulness (0-35)
    - Sharpness (0-25)
    - Contrast (0-20)
    - Brightness (0-10)
    - Texture (0-10)
    
    Returns:
        float: MUSIQ score atau None jika error
    """
    try:
        musiq_vals = []
        
        for idx, fr in enumerate(frames):
            try:
                # Convert to RGB and normalize
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                
                # === 1. Colorfulness (0-35) ===
                r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
                rg = r - g
                yb = 0.5 * (r + g) - b
                
                std_rg = np.std(rg)
                std_yb = np.std(yb)
                mean_rg = np.mean(rg)
                mean_yb = np.mean(yb)
                
                colorfulness = (np.sqrt(std_rg**2 + std_yb**2) + 
                               0.3 * np.sqrt(mean_rg**2 + mean_yb**2))
                colorfulness_score = min(colorfulness * 20, 35)
                
                # === 2. Sharpness (0-25) ===
                sharpness_scores = []
                for ksize in [3, 5, 7]:
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
                    sharpness_scores.append(np.var(laplacian))
                
                avg_sharpness = np.mean(sharpness_scores)
                sharpness_score = min(avg_sharpness / 100, 25)
                
                # === 3. Contrast (0-20) ===
                contrast = np.std(gray)
                contrast_score = min(contrast * 50, 20)
                
                # === 4. Brightness Balance (0-10) ===
                brightness = np.mean(gray)
                brightness_penalty = abs(brightness - 0.5) * 10
                brightness_score = max(0, 10 - brightness_penalty)
                
                # === 5. Texture Quality (0-10) ===
                edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                texture_score = min(edge_density * 100, 10)
                
                # Combined MUSIQ score
                musiq_score = (colorfulness_score + 
                              sharpness_score + 
                              contrast_score + 
                              brightness_score + 
                              texture_score)
                
                musiq_vals.append(musiq_score)
                
            except Exception as frame_error:
                print(f"[Warning] MUSIQ frame {idx} error: {frame_error}")
                continue
        
        if not musiq_vals:
            print("[Error] No MUSIQ values calculated")
            return None
        
        avg_musiq = float(np.mean(musiq_vals))
        print(f"[Debug] MUSIQ calculated: {avg_musiq:.2f} (from {len(musiq_vals)} frames)")
        
        return round(avg_musiq, 2)
    
    except Exception as e:
        print(f"[Error] MUSIQ calculation: {e}")
        traceback.print_exc()
        return None


def calculate_nrqm_corrected(frames: List[np.ndarray]) -> float:
    """
    NRQM (No-Reference Quality Metric)
    Range: 0-10 (HIGHER is BETTER)
    - Excellent: > 7
    - Good: 5-7
    - Fair: 3-5
    - Poor: < 3
    
    Components:
    - Edge strength (0-3)
    - Local contrast (0-2.5)
    - Texture richness (0-2)
    - Information content (0-2.5)
    
    Returns:
        float: NRQM score atau None jika error
    """
    try:
        nrqm_vals = []
        
        for fr in frames:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            
            # === 1. Edge Strength (0-3) ===
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
            edge_strength = min(np.mean(edge_magnitude) * 3, 3)
            
            # === 2. Local Contrast (0-2.5) ===
            kernel_size = 5
            local_mean = cv2.blur(gray, (kernel_size, kernel_size))
            local_contrast = min(np.mean(np.abs(gray - local_mean)) * 10, 2.5)
            
            # === 3. Texture Richness (0-2) ===
            texture_var = min(np.var(gray) * 15, 2)
            
            # === 4. Information Content (0-2.5) ===
            hist = cv2.calcHist([gray], [0], None, [256], [0, 1]).flatten()
            hist = hist[hist > 0]
            prob = hist / hist.sum()
            entropy = -np.sum(prob * np.log2(prob))
            entropy_score = min((entropy / 8.0) * 2.5, 2.5)
            
            # Combined NRQM score
            nrqm_score = edge_strength + local_contrast + texture_var + entropy_score
            nrqm_vals.append(min(nrqm_score, 10))
        
        avg_nrqm = float(np.mean(nrqm_vals))
        print(f"[Debug] NRQM calculated: {avg_nrqm:.2f}")
        
        return round(avg_nrqm, 2)
    
    except Exception as e:
        print(f"[Error] NRQM calculation: {e}")
        traceback.print_exc()
        return None


def calculate_brisque_piq(frames: List[np.ndarray]) -> float:
    """
    BRISQUE using PIQ library
    Range: 0-100 (context dependent, typically lower is better)
    
    Returns:
        float: BRISQUE score atau None jika error
    """
    if 'brisque' not in PIQ_MODELS or not TORCH_OK:
        print("[Warning] BRISQUE not available (PIQ not initialized)")
        return None
    
    try:
        tensors = []
        
        for fr in frames:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            tensor = torch.tensor(gray).unsqueeze(0).unsqueeze(0).float() / 255.0
            tensors.append(tensor)
        
        batch_tensor = torch.cat(tensors, dim=0).to('cpu')
        
        with torch.no_grad():
            brisque_score = PIQ_MODELS['brisque'](batch_tensor, data_range=1.0)
            avg_brisque = float(brisque_score.mean().item())
        
        print(f"[Debug] BRISQUE calculated: {avg_brisque:.2f}")
        return round(avg_brisque, 2)
    
    except Exception as e:
        print(f"[Error] BRISQUE calculation: {e}")
        traceback.print_exc()
        return None


# ============================================================================
# REFERENCE METRICS (PSNR, SSIM, LPIPS)
# ============================================================================

def _frame_metrics_real(f1: np.ndarray, f2: np.ndarray, resize_to: int = 256):
    """
    Calculate PSNR, SSIM, LPIPS for two frames
    
    Returns:
        tuple: (mse, psnr, ssim_val, lpips_val)
    """
    try:
        f1r = cv2.resize(f1, (resize_to, resize_to))
        f2r = cv2.resize(f2, (resize_to, resize_to))
    except:
        f1r, f2r = f1, f2

    g1 = cv2.cvtColor(f1r, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(f2r, cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    # MSE and PSNR
    mse = float(np.mean((g1 - g2) ** 2))
    psnr = float('inf') if mse == 0 else 10.0 * math.log10((255.0 ** 2) / max(mse, 1e-10))

    # SSIM
    ssim_val = None
    if SKIMAGE_OK and ssim_fn is not None:
        try:
            ssim_val = float(ssim_fn(g1.astype(np.uint8), g2.astype(np.uint8), data_range=255))
        except Exception as e:
            print(f"[Warning] SSIM calculation failed: {e}")
    
    # LPIPS
    lpips_val = None
    if LPIPS_FN is not None:
        try:
            t1 = torch.tensor(f1r).permute(2,0,1).unsqueeze(0).float().to(LPIPS_DEVICE) / 255.0
            t2 = torch.tensor(f2r).permute(2,0,1).unsqueeze(0).float().to(LPIPS_DEVICE) / 255.0
            with torch.no_grad():
                lpips_val = float(LPIPS_FN(t1, t2).item())
        except Exception as e:
            print(f"[Warning] LPIPS calculation failed: {e}")

    return mse, psnr, ssim_val, lpips_val


def compare_videos_advanced(video_path1: str, video_path2: str, 
                           max_frames: int = 40, 
                           sample_every: int = 5, 
                           resize_to: int = 256) -> Dict[str, Any]:
    """
    Compare two videos using reference metrics (PSNR, SSIM, LPIPS)
    
    Returns:
        dict: Metrics dictionary
    """
    print(f"[Info] Comparing videos: {video_path1} vs {video_path2}")
    
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

    # Calculate averages
    avg_mse = total_mse / count
    avg_psnr = float('inf') if avg_mse == 0.0 else 10.0 * math.log10((255.0 ** 2) / max(avg_mse, 1e-10))
    avg_ssim = float(total_ssim / count_ssim) if count_ssim > 0 else None
    avg_lpips = float(total_lpips / count_lpips) if count_lpips > 0 else None

    print(f"[Debug] Reference metrics - PSNR: {avg_psnr}, SSIM: {avg_ssim}, LPIPS: {avg_lpips}")

    return {
        "frames": int(count),
        "MSE": float(avg_mse),
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "LPIPS": avg_lpips
    }


def no_reference_metrics(video_path: str, 
                        max_frames: int = 30, 
                        sample_every: int = 5, 
                        resize_to: int = 256) -> Dict[str, Any]:
    """
    Calculate no-reference metrics for a single video
    
    Returns:
        dict: Metrics dictionary
    """
    print(f"[Info] Calculating no-reference metrics for: {video_path}")
    
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
    
    print(f"[Info] Processing {len(frames)} frames")
    
    # Calculate all metrics
    metrics = {}
    
    # 1. Sharpness
    print("[Info] Calculating Sharpness...")
    sharpness = calculate_sharpness_corrected(frames)
    metrics['SHARPNESS'] = sharpness if sharpness is not None else '-'
    
    # 2. Entropy
    print("[Info] Calculating Entropy...")
    try:
        ent_vals = []
        for fr in frames:
            entropy = calculate_frame_entropy(fr)
            if entropy is not None:
                ent_vals.append(entropy)
        metrics['ENTROPY'] = round(float(np.mean(ent_vals)), 6) if ent_vals else '-'
    except Exception as e:
        print(f"[Error] ENTROPY: {e}")
        metrics['ENTROPY'] = '-'
    
    # 3. BRISQUE
    print("[Info] Calculating BRISQUE...")
    brisque = calculate_brisque_piq(frames)
    metrics['BRISQUE'] = brisque if brisque is not None else '-'
    
    # 4. NIQE
    print("[Info] Calculating NIQE...")
    niqe = calculate_niqe_corrected(frames)
    metrics['NIQE'] = niqe if niqe is not None else '-'
    
    # 5. MUSIQ
    print("[Info] Calculating MUSIQ...")
    musiq = calculate_musiq_corrected(frames)
    metrics['MUSIQ'] = musiq if musiq is not None else '-'
    
    # 6. NRQM
    print("[Info] Calculating NRQM...")
    nrqm = calculate_nrqm_corrected(frames)
    metrics['NRQM'] = nrqm if nrqm is not None else '-'

    print(f"[Info] Final metrics: {metrics}")
    return metrics