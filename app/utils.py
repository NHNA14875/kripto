# app/utils.py
import os
import math
import numpy as np
import datetime
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend untuk thread-safety

# --- Fungsi Histogram dan Entropy ---

def file_byte_histogram(path):
    """Menghitung histogram byte dari file"""
    hist = np.zeros(256, dtype=np.int64)
    with open(path, "rb") as f:
        while True:
            b = f.read(65536)
            if not b:
                break
            arr = np.frombuffer(b, dtype=np.uint8)
            if arr.size:
                counts = np.bincount(arr, minlength=256)
                hist[:counts.size] += counts
    return hist.tolist()

def compute_entropy_from_hist(hist):
    """Menghitung entropy dari histogram"""
    counts = np.array(hist, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return None
    p = counts[counts > 0] / total
    ent = -(p * np.log2(p)).sum()
    return round(float(ent), 6)  # 6 desimal untuk presisi lebih tinggi

def calculate_file_entropy(file_path):
    """
    Menghitung entropy file secara langsung
    Wrapper untuk kemudahan
    """
    try:
        if not file_path or not os.path.exists(file_path):
            return None
        hist = file_byte_histogram(file_path)
        return compute_entropy_from_hist(hist)
    except Exception as e:
        print(f"[Error] Gagal menghitung entropy untuk {file_path}: {e}")
        return None

# --- Fungsi Helper ---

def _filesize_kb(path):
    """Menghitung ukuran file dalam KB"""
    try:
        if path and os.path.exists(path):
            return round(os.path.getsize(path) / 1024.0, 1)
    except Exception:
        pass
    return "-"

def format_psnr_desc(psnr_value, mse_value):
    """Return (display_val, desc_string). display_val numeric or '-'."""
    if isinstance(psnr_value, float) and math.isinf(psnr_value):
        return float('inf'), "Tak Terhingga (MSE=0)"
    try:
        pv = float(psnr_value)
    except Exception:
        return "-", "-"
    
    tag = "distorsi tinggi"
    if pv > 50: tag = "nyaris sempurna"
    elif pv > 40: tag = "distorsi rendah"
    elif pv > 30: tag = "distorsi sedang"
    
    desc = f"{pv:.2f} dB ({tag})"
    return pv, desc

def format_metric_value(key, value):
    """Format nilai metrik untuk tampilan yang konsisten"""
    if value is None or value == "-":
        return "-"
    
    if isinstance(value, str):
        return value
    
    if isinstance(value, float):
        if key == "PSNR" and math.isinf(value):
            return "inf"
        elif key in ["SSIM", "LPIPS"]:
            return f"{value:.6f}"
        elif key in ["FileEntropy"]:
            return f"{value:.6f}"
        elif key in ["UkuranKB"]:
            return f"{value:.1f}"
        else:
            return f"{value:.3f}"
    
    return str(value)

def create_histogram_comparison(paths, out_dir):
    """
    Membuat histogram perbandingan untuk 3 file:
    - Video Asli
    - File Terenkripsi
    - Video Dekripsi
    
    Returns: path ke file histogram PNG
    """
    try:
        print("[Info] Membuat histogram perbandingan...")
        
        # Siapkan figure dengan 3 subplot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Histogram Perbandingan Byte Distribution', fontsize=16, fontweight='bold')
        
        # Data untuk setiap file
        files_data = [
            {"path": paths.get("orig"), "title": "Video Asli", "color": "#4CAF50"},
            {"path": paths.get("enc"), "title": "File Terenkripsi", "color": "#F44336"},
            {"path": paths.get("dec"), "title": "Video Dekripsi", "color": "#2196F3"}
        ]
        
        for idx, data in enumerate(files_data):
            ax = axes[idx]
            file_path = data["path"]
            
            if not file_path or not os.path.exists(file_path):
                ax.text(0.5, 0.5, 'File tidak tersedia', ha='center', va='center')
                ax.set_title(data["title"])
                continue
            
            # Hitung histogram
            hist = file_byte_histogram(file_path)
            entropy = compute_entropy_from_hist(hist)
            
            # Plot histogram
            x = np.arange(256)
            ax.bar(x, hist, color=data["color"], alpha=0.7, width=1.0)
            
            # Styling
            ax.set_title(f'{data["title"]}\nEntropy: {entropy:.6f}', fontweight='bold')
            ax.set_xlabel('Byte Value (0-255)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 255)
            
            # Format y-axis dengan scientific notation untuk angka besar
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        # Simpan histogram
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        histogram_path = out_dir / f"histogram_comparison_{timestamp}.png"
        
        plt.savefig(histogram_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"[Info] Histogram disimpan di: {histogram_path}")
        return str(histogram_path)
    
    except Exception as e:
        print(f"[Error] Gagal membuat histogram: {e}")
        import traceback
        traceback.print_exc()
        return None
    """Format nilai metrik untuk tampilan yang konsisten"""
    if value is None or value == "-":
        return "-"
    
    if isinstance(value, str):
        return value
    
    if isinstance(value, float):
        if key == "PSNR" and math.isinf(value):
            return "inf"
        elif key in ["SSIM", "LPIPS"]:
            return f"{value:.6f}"
        elif key in ["FileEntropy"]:
            return f"{value:.6f}"
        elif key in ["UkuranKB"]:
            return f"{value:.1f}"
        else:
            return f"{value:.3f}"
    
    return str(value)

# --- Fungsi Utama: Simpan Laporan ---

def save_metrics_summary(results_ref_dec, results_noref_orig, results_noref_dec, paths, out_dir):
    """
    Menyimpan ringkasan metrik ke CSV dan HTML
    
    DIPERBAIKI: Menghitung FileEntropy untuk semua file (termasuk video asli dan dekripsi)
    """
    print("[Info] Memulai pembuatan laporan metrik...")
    
    rows = []
    ts = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
    keys = ["Jenis","PSNR","SSIM","LPIPS","MUSIQ","NRQM","NIQE","BRISQUE","Sharpness","FileEntropy","UkuranKB","Timestamp"]

    # === 1. Video Asli ===
    print("[Info] Menghitung FileEntropy untuk video asli...")
    orig_entropy = calculate_file_entropy(paths.get("orig"))
    
    rows.append({
        "Jenis": "Asli",
        "PSNR": "-",
        "SSIM": "-",
        "LPIPS": "-",
        "MUSIQ": results_noref_orig.get("MUSIQ", "-"),
        "NRQM": results_noref_orig.get("NRQM", "-"),
        "NIQE": results_noref_orig.get("NIQE", "-"),
        "BRISQUE": results_noref_orig.get("BRISQUE", "-"),
        "Sharpness": results_noref_orig.get("SHARPNESS", "-"),
        "FileEntropy": orig_entropy if orig_entropy else "-",
        "UkuranKB": _filesize_kb(paths.get("orig")),
        "Timestamp": ts
    })
    print(f"[Debug] Entropy video asli: {orig_entropy}")

    # === 2. File Terenkripsi ===
    enc_entropy = "-"
    if paths.get("enc") and os.path.exists(paths.get("enc")):
        print("[Info] Menghitung FileEntropy untuk file terenkripsi...")
        enc_entropy = calculate_file_entropy(paths["enc"])
        print(f"[Debug] Entropy file terenkripsi: {enc_entropy}")
    
    rows.append({
        "Jenis": "Terenkripsi",
        "PSNR": "-",
        "SSIM": "-",
        "LPIPS": "-",
        "MUSIQ": "-",
        "NRQM": "-",
        "NIQE": "-",
        "BRISQUE": "-",
        "Sharpness": "-",
        "FileEntropy": enc_entropy if enc_entropy else "-",
        "UkuranKB": _filesize_kb(paths.get("enc")),
        "Timestamp": ts
    })

    # === 3. Video Dekripsi ===
    print("[Info] Menghitung FileEntropy untuk video dekripsi...")
    dec_entropy = calculate_file_entropy(paths.get("dec"))
    
    rows.append({
        "Jenis": "Didekripsi",
        "PSNR": results_ref_dec.get("PSNR", "-"),
        "SSIM": results_ref_dec.get("SSIM", "-"),
        "LPIPS": results_ref_dec.get("LPIPS", "-"),
        "MUSIQ": results_noref_dec.get("MUSIQ", "-"),
        "NRQM": results_noref_dec.get("NRQM", "-"),
        "NIQE": results_noref_dec.get("NIQE", "-"),
        "BRISQUE": results_noref_dec.get("BRISQUE", "-"),
        "Sharpness": results_noref_dec.get("SHARPNESS", "-"),
        "FileEntropy": dec_entropy if dec_entropy else "-",
        "UkuranKB": _filesize_kb(paths.get("dec")),
        "Timestamp": ts
    })
    print(f"[Debug] Entropy video dekripsi: {dec_entropy}")

    # === Simpan ke File ===
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"metrics_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    csv_path = out_dir / f"{prefix}.csv"
    html_path = out_dir / f"{prefix}.html"

    # === CSV ===
    print(f"[Info] Menyimpan CSV ke: {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            row_data = []
            for k in keys:
                val = r.get(k, "-")
                row_data.append(format_metric_value(k, val))
            w.writerow(row_data)

    # === HTML dengan Styling yang Lebih Baik ===
    print(f"[Info] Menyimpan HTML ke: {html_path}")
    html_lines = []
    html_lines.append("<!doctype html>")
    html_lines.append("<html>")
    html_lines.append("<head>")
    html_lines.append("<meta charset='utf-8'>")
    html_lines.append("<title>Ringkasan Metrik Video Kriptografi</title>")
    html_lines.append("<style>")
    html_lines.append("body { font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; background: #1a1a1a; color: #e0e0e0; }")
    html_lines.append("h1 { color: #4CAF50; text-align: center; }")
    html_lines.append("h3 { color: #2196F3; margin-top: 30px; }")
    html_lines.append("table { border-collapse: collapse; width: 100%; background: #2a2a2a; border: 2px solid #444; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }")
    html_lines.append("th, td { padding: 12px 15px; border: 1px solid #444; text-align: center; }")
    html_lines.append("th { background: #3a3a3a; color: #4CAF50; font-weight: bold; }")
    html_lines.append("tr:nth-child(even) { background: #252525; }")
    html_lines.append("tr:hover { background: #333; }")
    html_lines.append(".excellent { color: #4CAF50; font-weight: bold; }")
    html_lines.append(".good { color: #8BC34A; }")
    html_lines.append(".warning { color: #FFC107; }")
    html_lines.append(".info { color: #2196F3; }")
    html_lines.append(".footer { text-align: center; margin-top: 30px; color: #888; font-size: 0.9em; }")
    html_lines.append("</style>")
    html_lines.append("</head>")
    html_lines.append("<body>")
    html_lines.append("<h1>📊 Ringkasan Metrik Video Kriptografi</h1>")
    html_lines.append(f"<p style='text-align:center; color:#888;'>Generated: {ts}</p>")
    
    html_lines.append("<h3>📈 Tabel Metrik Lengkap</h3>")
    html_lines.append("<table>")
    html_lines.append("<thead><tr>")
    for k in keys:
        html_lines.append(f"<th>{k}</th>")
    html_lines.append("</tr></thead>")
    html_lines.append("<tbody>")
    
    for r in rows:
        html_lines.append("<tr>")
        for k in keys:
            v = r.get(k, "-")
            formatted_val = format_metric_value(k, v)
            
            # Styling khusus berdasarkan nilai
            css_class = ""
            if k == "PSNR" and formatted_val == "inf":
                css_class = "excellent"
                formatted_val = "∞ (Perfect)"
            elif k == "SSIM" and isinstance(v, float) and v >= 0.99:
                css_class = "excellent"
            elif k == "LPIPS" and isinstance(v, float) and v <= 0.01:
                css_class = "excellent"
            elif k == "FileEntropy" and isinstance(v, float) and v >= 7.5:
                css_class = "excellent"
                formatted_val += " (High)"
            
            html_lines.append(f"<td class='{css_class}'>{formatted_val}</td>")
        html_lines.append("</tr>")
    
    html_lines.append("</tbody>")
    html_lines.append("</table>")
    
    # Penjelasan Metrik
    html_lines.append("<h3>ℹ️ Penjelasan Metrik</h3>")
    html_lines.append("<table>")
    html_lines.append("<tr><th>Metrik</th><th>Deskripsi</th><th>Range Ideal</th></tr>")
    html_lines.append("<tr><td>PSNR</td><td>Peak Signal-to-Noise Ratio (dB)</td><td>&gt;40 dB (Excellent)</td></tr>")
    html_lines.append("<tr><td>SSIM</td><td>Structural Similarity Index</td><td>0.95-1.0 (Excellent)</td></tr>")
    html_lines.append("<tr><td>LPIPS</td><td>Learned Perceptual Similarity</td><td>0.0-0.1 (Lower is better)</td></tr>")
    html_lines.append("<tr><td>MUSIQ</td><td>Multi-Scale Image Quality</td><td>0-100 (Higher is better)</td></tr>")
    html_lines.append("<tr><td>NRQM</td><td>No-Reference Quality Metric</td><td>0-10 (Higher is better)</td></tr>")
    html_lines.append("<tr><td>NIQE</td><td>Natural Image Quality Evaluator</td><td>0-10 (Lower is better)</td></tr>")
    html_lines.append("<tr><td>BRISQUE</td><td>Blind Image Spatial Quality</td><td>0-100 (Context dependent)</td></tr>")
    html_lines.append("<tr><td>Sharpness</td><td>Laplacian Variance</td><td>Higher = More detail</td></tr>")
    html_lines.append("<tr><td>FileEntropy</td><td>Shannon Entropy (bits)</td><td>7.5-8.0 (Encrypted files)</td></tr>")
    html_lines.append("</table>")
    
    html_lines.append("<div class='footer'>")
    html_lines.append("<p>Generated by Video Crypto Modern | Blowfish + RSA Encryption</p>")
    html_lines.append("</div>")
    html_lines.append("</body>")
    html_lines.append("</html>")
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))
    
    # === Generate Histogram ===
    histogram_path = create_histogram_comparison(paths, out_dir)
    
    print("[Info] Laporan berhasil disimpan!")
    return {
        "csv": str(csv_path), 
        "html": str(html_path),
        "histogram": histogram_path
    }