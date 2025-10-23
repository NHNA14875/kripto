# app/metrics.py

import cv2
import numpy as np
import time
from app.metrics_core import psnr, calculate_ssim, calculate_brisque, calculate_nrqm, calculate_musiq
from app.utils import format_elapsed_time

def evaluate_metrics(original_img_path, decrypted_img_path, output_callback=print):
    """
    Menghitung dan mencetak metrik kualitas gambar antara gambar asli dan dekripsi.

    Args:
        original_img_path (str): Path ke gambar asli.
        decrypted_img_path (str): Path ke gambar hasil dekripsi.
        output_callback (function): Fungsi untuk menampilkan output (default: print).
    """
    try:
        # Baca gambar
        start_read = time.time()
        img_asli = cv2.imread(original_img_path)
        img_dekrip = cv2.imread(decrypted_img_path)
        read_time = time.time() - start_read

        if img_asli is None:
            output_callback(f"🔴 Error: Gagal membaca gambar asli: {original_img_path}")
            return False, {}
        if img_dekrip is None:
            output_callback(f"🔴 Error: Gagal membaca gambar dekripsi: {decrypted_img_path}")
            return False, {}

        output_callback(f"⏱️ Waktu membaca gambar: {format_elapsed_time(read_time)}")

        metrics_results = {}
        all_metrics_successful = True

        # 1. Perbandingan Langsung (Identik Piksel)
        output_callback("\n--- Perbandingan Piksel ---")
        start_pixel = time.time()
        if img_asli.shape == img_dekrip.shape and not(np.bitwise_xor(img_asli,img_dekrip).any()):
            output_callback("✅ Piksel Identik (OK)")
            metrics_results['pixel_identical'] = True
        else:
            output_callback("🔴 Piksel TIDAK Identik!")
            all_metrics_successful = False
            metrics_results['pixel_identical'] = False
        pixel_time = time.time() - start_pixel
        output_callback(f"⏱️ Waktu perbandingan piksel: {format_elapsed_time(pixel_time)}")

        # 2. PSNR (Peak Signal-to-Noise Ratio)
        output_callback("\n--- PSNR ---")
        start_psnr = time.time()
        try:
            psnr_val = psnr(img_asli, img_dekrip)
            output_callback(f"PSNR = {psnr_val:.2f}")
            metrics_results['psnr'] = psnr_val
            # Tidak ada lagi pengecekan ambang batas, hanya perhitungan
        except ValueError as e:
            output_callback(f"🔴 Error PSNR: {e}")
            all_metrics_successful = False
            metrics_results['psnr'] = np.nan
        except Exception as e:
            output_callback(f"🔴 Error PSNR tidak terduga: {e}")
            all_metrics_successful = False
            metrics_results['psnr'] = np.nan
        psnr_time = time.time() - start_psnr
        output_callback(f"⏱️ Waktu perhitungan PSNR: {format_elapsed_time(psnr_time)}")


        # 3. SSIM (Structural Similarity Index Measure)
        output_callback("\n--- SSIM ---")
        start_ssim = time.time()
        try:
            ssim_val = calculate_ssim(img_asli, img_dekrip)
            output_callback(f"SSIM = {ssim_val:.4f}")
            metrics_results['ssim'] = ssim_val
            if not np.isclose(ssim_val, 1.0):
                 output_callback("⚠️ Nilai SSIM tidak 1.0 (meskipun piksel mungkin identik, perhitungan bisa berbeda karena float precision atau konversi grayscale)")
                 # Tidak dianggap error fatal jika piksel identik
        except ValueError as e:
            output_callback(f"🔴 Error SSIM: {e}")
            all_metrics_successful = False
            metrics_results['ssim'] = np.nan
        except Exception as e:
            output_callback(f"🔴 Error SSIM tidak terduga: {e}")
            all_metrics_successful = False
            metrics_results['ssim'] = np.nan
        ssim_time = time.time() - start_ssim
        output_callback(f"⏱️ Waktu perhitungan SSIM: {format_elapsed_time(ssim_time)}")

        # 4. BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
        output_callback("\n--- BRISQUE ---")
        start_brisque = time.time()
        brisque_asli = calculate_brisque(original_img_path)
        brisque_dekrip = calculate_brisque(decrypted_img_path)
        brisque_time = time.time() - start_brisque
        output_callback(f"⏱️ Waktu perhitungan BRISQUE: {format_elapsed_time(brisque_time)}")

        if not np.isnan(brisque_asli) and not np.isnan(brisque_dekrip):
            output_callback(f"BRISQUE Asli = Didekripsi: {brisque_asli:.2f} = {brisque_dekrip:.2f}", end="")
            if np.isclose(brisque_asli, brisque_dekrip):
                output_callback(" Harus identik (OK)")
                # ----- PERINGATAN DIKOMENTARI -----
                # if brisque_asli > 50: # Contoh ambang batas
                #    output_callback(" \U0001F534 Terlalu tinggi!", end="")
                # ------------------------------------
                metrics_results['brisque_asli'] = brisque_asli
                metrics_results['brisque_dekrip'] = brisque_dekrip
                metrics_results['brisque_match'] = True
            else:
                output_callback(" \U0001F534 TIDAK IDENTIK!", end="")
                all_metrics_successful = False
                metrics_results['brisque_match'] = False
            output_callback() # Pindah baris
        else:
             output_callback("🔴 Gagal menghitung salah satu atau kedua nilai BRISQUE.")
             all_metrics_successful = False
             metrics_results['brisque_match'] = False


        # 5. NRQM (Naturalness Related Quality Metric - Implementasi dari metrics_core)
        output_callback("\n--- NRQM ---")
        start_nrqm = time.time()
        nrqm_asli = calculate_nrqm(original_img_path)
        nrqm_dekrip = calculate_nrqm(decrypted_img_path)
        nrqm_time = time.time() - start_nrqm
        output_callback(f"⏱️ Waktu perhitungan NRQM: {format_elapsed_time(nrqm_time)}")

        if not np.isnan(nrqm_asli) and not np.isnan(nrqm_dekrip):
            output_callback(f"NRQM Asli = Didekripsi: {nrqm_asli:.2f} = {nrqm_dekrip:.2f}", end="")
            if np.isclose(nrqm_asli, nrqm_dekrip):
                output_callback(" Harus identik (OK)")
                # ----- PERINGATAN DIKOMENTARI -----
                # if nrqm_asli < 3: # Contoh ambang batas
                #     output_callback(" \U0001F534 Terlalu rendah!", end="")
                # ------------------------------------
                metrics_results['nrqm_asli'] = nrqm_asli
                metrics_results['nrqm_dekrip'] = nrqm_dekrip
                metrics_results['nrqm_match'] = True
            else:
                output_callback(" \U0001F534 TIDAK IDENTIK!", end="")
                all_metrics_successful = False
                metrics_results['nrqm_match'] = False
            output_callback() # Pindah baris
        else:
             output_callback("🔴 Gagal menghitung salah satu atau kedua nilai NRQM.")
             all_metrics_successful = False
             metrics_results['nrqm_match'] = False

        # 6. MUSIQ (Multi-Scale Image Quality Transformer)
        output_callback("\n--- MUSIQ ---")
        start_musiq = time.time()
        musiq_asli = calculate_musiq(original_img_path)
        musiq_dekrip = calculate_musiq(decrypted_img_path)
        musiq_time = time.time() - start_musiq
        output_callback(f"⏱️ Waktu perhitungan MUSIQ: {format_elapsed_time(musiq_time)}")

        if not np.isnan(musiq_asli) and not np.isnan(musiq_dekrip):
             output_callback(f"MUSIQ Asli = Didekripsi: {musiq_asli:.2f} = {musiq_dekrip:.2f}", end="")
             if np.isclose(musiq_asli, musiq_dekrip):
                 output_callback(" Harus identik (OK)")
                 # ----- PERINGATAN DIKOMENTARI -----
                 # if musiq_asli < 20: # Contoh ambang batas
                 #    output_callback(" ⚠️ Nilai terlalu rendah", end="")
                 # ------------------------------------
                 metrics_results['musiq_asli'] = musiq_asli
                 metrics_results['musiq_dekrip'] = musiq_dekrip
                 metrics_results['musiq_match'] = True
             else:
                 output_callback(" \U0001F534 TIDAK IDENTIK!", end="")
                 all_metrics_successful = False
                 metrics_results['musiq_match'] = False
             output_callback() # Pindah baris
        else:
             output_callback("🔴 Gagal menghitung salah satu atau kedua nilai MUSIQ.")
             all_metrics_successful = False
             metrics_results['musiq_match'] = False


        output_callback("\n--- Kesimpulan ---")
        if all_metrics_successful:
            output_callback("✅ Semua metrik perbandingan (Piksel, BRISQUE, NRQM, MUSIQ) menunjukkan hasil identik.")
        else:
            output_callback("🔴 Terdapat perbedaan pada satu atau lebih metrik perbandingan atau terjadi error perhitungan.")

        return all_metrics_successful, metrics_results

    except Exception as e:
        output_callback(f"🔴 Terjadi error tidak terduga saat evaluasi metrik: {e}")
        import traceback
        output_callback(traceback.format_exc()) # Cetak traceback untuk debug
        return False, {}

# Contoh penggunaan jika file ini dijalankan langsung
if __name__ == '__main__':
    # Ganti dengan path gambar yang sesuai
    path_gambar_asli = 'gambar_asli.png'
    path_gambar_dekripsi = 'gambar_dekripsi.png'

    print(f"Membandingkan '{path_gambar_asli}' dengan '{path_gambar_dekripsi}'...")

    # Pastikan gambar contoh ada atau buat gambar dummy
    try:
        img_test = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(path_gambar_asli, img_test)
        cv2.imwrite(path_gambar_dekripsi, img_test)
        print("Membuat gambar dummy untuk pengujian...")
    except Exception as e:
        print(f"Tidak dapat membuat gambar dummy: {e}")

    sukses, hasil = evaluate_metrics(path_gambar_asli, path_gambar_dekripsi)

    if sukses:
        print("\nEvaluasi metrik berhasil.")
    else:
        print("\nEvaluasi metrik menemukan perbedaan atau error.")

    print("\nHasil Detail:")
    import json
    print(json.dumps(hasil, indent=2))