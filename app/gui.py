# app/gui.py
import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import traceback
import os
import time
import datetime
import webbrowser
from pathlib import Path
from PIL import Image, ImageTk

# Impor logika inti
from .crypto_core import encrypt_file, decrypt_file
from .metrics_core import compare_videos_advanced, no_reference_metrics
from .utils import save_metrics_summary

# Pengaturan Tampilan
ctk.set_appearance_mode("System")  # "Dark", "Light", "System"
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Video Crypto Modern (Blowfish + RSA)")
        self.geometry("900x650")
        self.minsize(700, 500)
        
        base_path = Path(__file__).resolve().parent.parent
        
        # --- Variabel ---
        self.input_path = ctk.StringVar()
        self.output_path = ctk.StringVar()
        self.pubkey_path = ctk.StringVar(value=str(base_path / "keys" / "rsa_public.pem"))
        self.privkey_path = ctk.StringVar(value=str(base_path / "keys" / "rsa_private.pem"))
        self.is_running = False # Flag untuk mencegah double-click

        # --- Layout ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0) # Frame Path
        self.grid_rowconfigure(1, weight=0) # Frame Aksi
        self.grid_rowconfigure(2, weight=1) # Log
        
        self._build_path_frame()
        self._build_action_frame()
        self._build_log_frame()
        self.log("Aplikasi dimulai. Siap.")

    def _build_path_frame(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)

        # Input
        ctk.CTkLabel(frame, text="Input (Video/VENC):").grid(row=0, column=0, padx=10, pady=(10,5), sticky="w")
        e1 = ctk.CTkEntry(frame, textvariable=self.input_path)
        e1.grid(row=0, column=1, padx=5, pady=(10,5), sticky="ew")
        b1 = ctk.CTkButton(frame, text="Browse...", width=100, command=self.browse_input)
        b1.grid(row=0, column=2, padx=(5,10), pady=(10,5))

        # Output
        ctk.CTkLabel(frame, text="Output (VENC/Video):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        e2 = ctk.CTkEntry(frame, textvariable=self.output_path)
        e2.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        b2 = ctk.CTkButton(frame, text="Browse...", width=100, command=self.browse_output)
        b2.grid(row=1, column=2, padx=(5,10), pady=5)

        # Public Key
        ctk.CTkLabel(frame, text="Public Key (Enkripsi):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        e3 = ctk.CTkEntry(frame, textvariable=self.pubkey_path)
        e3.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        b3 = ctk.CTkButton(frame, text="Browse...", width=100, command=lambda: self.browse_key(self.pubkey_path, "Public Key"))
        b3.grid(row=2, column=2, padx=(5,10), pady=5)

        # Private Key
        ctk.CTkLabel(frame, text="Private Key (Dekripsi):").grid(row=3, column=0, padx=10, pady=(5,10), sticky="w")
        e4 = ctk.CTkEntry(frame, textvariable=self.privkey_path)
        e4.grid(row=3, column=1, padx=5, pady=(5,10), sticky="ew")
        b4 = ctk.CTkButton(frame, text="Browse...", width=100, command=lambda: self.browse_key(self.privkey_path, "Private Key"))
        b4.grid(row=3, column=2, padx=(5,10), pady=(5,10))

    def _build_action_frame(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=0)
        frame.grid_columnconfigure((0,1,2), weight=1)

        # Tombol Aksi
        self.encrypt_btn = ctk.CTkButton(frame, text="Enkripsi", height=40, command=self.on_encrypt)
        self.encrypt_btn.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.decrypt_btn = ctk.CTkButton(frame, text="Dekripsi", height=40, command=self.on_decrypt)
        self.decrypt_btn.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.metrics_btn = ctk.CTkButton(frame, text="Uji Lanjutan (MSE, PSNR, Lanjutan)", height=40, fg_color="green", hover_color="darkgreen", command=self.on_advanced_test)
        self.metrics_btn.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

    def _build_log_frame(self):
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(5,10))
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.log_text = ctk.CTkTextbox(frame, wrap="none")
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=(0,5))
        
        scroll_y = ctk.CTkScrollbar(frame, command=self.log_text.yview)
        scroll_y.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll_y.set)

        self.progress = ctk.CTkProgressBar(frame, mode="determinate")
        self.progress.set(0)
        self.progress.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5,0))

    # --- Fungsi Logging dan Status ---
    
    def log(self, message):
        # Fungsi ini aman dipanggil dari thread lain
        def _insert():
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self.log_text.insert("end", f"[{ts}] {message}\n")
            self.log_text.see("end")
        self.after(0, _insert)

    def set_progress(self, value):
        # Aman dipanggil dari thread
        self.after(0, lambda: self.progress.set(value))

    def start_indeterminate(self):
        self.after(0, lambda: (
            self.progress.configure(mode="indeterminate"),
            self.progress.start()
        ))

    def stop_progress(self):
        self.after(0, lambda: (
            self.progress.stop(),
            self.progress.configure(mode="determinate"),
            self.progress.set(0)
        ))

    def set_running(self, is_running):
        self.is_running = is_running
        state = "disabled" if is_running else "normal"
        self.after(0, lambda: (
            self.encrypt_btn.configure(state=state),
            self.decrypt_btn.configure(state=state),
            self.metrics_btn.configure(state=state)
        ))

    # --- Fungsi File Dialog ---

    def browse_input(self):
        p = filedialog.askopenfilename(title="Pilih file input", filetypes=[("Video/Encrypted","*.mp4;*.mkv;*.avi;*.venc"),("All","*.*")])
        if not p: return
        
        self.input_path.set(p)
        self.log(f"Input diatur ke: {p}")
        
        # Otomatis sarankan output
        if p.lower().endswith(".venc"):
            self.output_path.set(os.path.splitext(p)[0] + "_DECRYPTED.mp4")
        else:
            self.output_path.set(p + ".venc")
        self.log(f"Output disarankan: {self.output_path.get()}")

    def browse_output(self):
        p = filedialog.asksaveasfilename(title="Simpan sebagai")
        if p:
            self.output_path.set(p)
            self.log(f"Output diatur ke: {p}")

    def browse_key(self, str_var, title):
        p = filedialog.askopenfilename(title=f"Pilih {title}", filetypes=[("PEM Files","*.pem")])
        if p:
            str_var.set(p)
            self.log(f"{title} diatur ke: {p}")
            
    # --- Fungsi Aksi (Tombol) ---
    
    def run_task(self, task_function, *args):
        if self.is_running:
            self.log("ERROR: Operasi lain sedang berjalan.")
            return
        
        def worker():
            self.set_running(True)
            try:
                task_function(*args)
            except Exception as e:
                self.log(f"!!! ERROR KRITIS !!!\n{traceback.format_exc()}")
                messagebox.showerror("Error", f"Terjadi error: {e}")
            finally:
                self.set_running(False)
                self.stop_progress()

        threading.Thread(target=worker, daemon=True).start()

    def on_encrypt(self):
        in_p = self.input_path.get()
        out_p = self.output_path.get()
        pub = self.pubkey_path.get()
        if not all([in_p, out_p, pub]):
            messagebox.showwarning("Input Kurang", "Harap isi path Input, Output, dan Public Key.")
            return
        
        def task():
            self.log(f"Mulai enkripsi: {in_p} -> {out_p}")
            self.start_indeterminate() # Pakai indeterminate untuk crypto
            
            start_time = time.time()
            encrypt_file(in_p, out_p, pub, progress_cb=None) # Hapus progress_cb jika tidak ada
            duration = time.time() - start_time
            
            self.stop_progress()
            self.log(f"Enkripsi Selesai (durasi {duration:.2f} detik). Output: {out_p}")
            messagebox.showinfo("Selesai", f"Enkripsi selesai!\nOutput: {out_p}")
        
        self.run_task(task)

    def on_decrypt(self):
        in_p = self.input_path.get()
        out_p = self.output_path.get()
        priv = self.privkey_path.get()
        if not all([in_p, out_p, priv]):
            messagebox.showwarning("Input Kurang", "Harap isi path Input, Output, dan Private Key.")
            return

        def task():
            self.log(f"Mulai dekripsi: {in_p} -> {out_p}")
            self.start_indeterminate()
            
            start_time = time.time()
            decrypt_file(in_p, out_p, priv, progress_cb=None) # Hapus progress_cb jika tidak ada
            duration = time.time() - start_time
            
            self.stop_progress()
            self.log(f"Dekripsi Selesai (durasi {duration:.2f} detik). Output: {out_p}")
            messagebox.showinfo("Selesai", f"Dekripsi selesai!\nOutput: {out_p}")

        self.run_task(task)

    def on_advanced_test(self):
        """
        DIPERBAIKI: Proper error handling dan logging untuk metrik
        """
        self.log("Memulai Uji Lanjutan...")
        orig = filedialog.askopenfilename(title="Pilih Video ASLI (Original)", filetypes=[("Video","*.mp4;*.mkv;*.avi")])
        if not orig:
            self.log("Uji dibatalkan (tidak ada video asli)."); return

        enc = filedialog.askopenfilename(title="Pilih File TERENKRIPSI (.venc) (Opsional)", filetypes=[("Encrypted","*.venc")])
        if not enc: enc = None # Opsional
            
        dec = filedialog.askopenfilename(title="Pilih Video DEKRIPSI (Decrypted)", filetypes=[("Video","*.mp4;*.mkv;*.avi")])
        if not dec:
            self.log("Uji dibatalkan (tidak ada video dekripsi)."); return
        
        paths = {"orig": orig, "enc": enc, "dec": dec}
        out_folder = os.path.dirname(orig)
        
        def task():
            start_time = time.time()
            self.set_running(True)
            
            try:
                # 1. Reference Metrics
                self.log("=" * 60)
                self.log("Menghitung (1/3): Reference Metrics (PSNR, SSIM, LPIPS)...")
                self.set_progress(0.1)
                
                try:
                    ref = compare_videos_advanced(orig, dec)
                    self.log(f"✓ PSNR: {ref.get('PSNR', '-')}")
                    self.log(f"✓ SSIM: {ref.get('SSIM', '-')}")
                    self.log(f"✓ LPIPS: {ref.get('LPIPS', '-')}")
                except Exception as e:
                    self.log(f"✗ ERROR Reference Metrics: {e}")
                    self.log(traceback.format_exc())
                    ref = {"PSNR": "-", "SSIM": "-", "LPIPS": "-"}

                # 2. No-Reference Asli
                self.log("=" * 60)
                self.log("Menghitung (2/3): No-Reference Metrics (Video Asli)...")
                self.set_progress(0.4)
                
                try:
                    noref_orig = no_reference_metrics(orig)
                    self.log(f"[GUI DEBUG] Hasil metrik ASLI: {noref_orig}")
                    
                    # Log setiap metrik
                    for key, val in noref_orig.items():
                        self.log(f"✓ {key} (Asli): {val}")
                except Exception as e:
                    self.log(f"✗ ERROR No-Ref Asli: {e}")
                    self.log(traceback.format_exc())
                    noref_orig = {k: "-" for k in ['SHARPNESS', 'ENTROPY', 'BRISQUE', 'NIQE', 'MUSIQ', 'NRQM']}

                # 3. No-Reference Dekripsi
                self.log("=" * 60)
                self.log("Menghitung (3/3): No-Reference Metrics (Video Dekripsi)...")
                self.set_progress(0.7)
                
                try:
                    noref_dec = no_reference_metrics(dec)
                    self.log(f"[GUI DEBUG] Hasil metrik DEKRIPSI: {noref_dec}")
                    
                    # Log setiap metrik
                    for key, val in noref_dec.items():
                        self.log(f"✓ {key} (Dekripsi): {val}")
                except Exception as e:
                    self.log(f"✗ ERROR No-Ref Dekripsi: {e}")
                    self.log(traceback.format_exc())
                    noref_dec = {k: "-" for k in ['SHARPNESS', 'ENTROPY', 'BRISQUE', 'NIQE', 'MUSIQ', 'NRQM']}

                # 4. Simpan Laporan
                self.log("=" * 60)
                self.log("Membuat laporan HTML dan CSV...")
                self.set_progress(0.9)
                
                try:
                    report_paths = save_metrics_summary(ref, noref_orig, noref_dec, paths, out_folder)
                    self.log(f"✓ Laporan disimpan di: {report_paths['html']}")
                except Exception as e:
                    self.log(f"✗ ERROR Simpan Laporan: {e}")
                    self.log(traceback.format_exc())
                    report_paths = None
                
                self.stop_progress()
                duration = time.time() - start_time
                self.log("=" * 60)
                self.log(f"✓ UJI LANJUTAN SELESAI (Durasi: {duration:.2f} detik)")
                self.log("=" * 60)

                if report_paths and messagebox.askyesno("Uji Selesai", f"Uji metrik lanjutan selesai.\n\nLaporan disimpan sebagai:\n{report_paths['html']}\n\nBuka laporan di browser?"):
                    try:
                        webbrowser.open_new_tab("file://" + os.path.abspath(report_paths["html"]))
                    except Exception as e_web:
                        self.log(f"Gagal membuka browser: {e_web}")
            
            except Exception as e:
                self.log(f"!!! ERROR KRITIS ADVANCED TEST !!!")
                self.log(traceback.format_exc())
                messagebox.showerror("Error", f"Terjadi error: {e}")
            
            finally:
                self.set_running(False)
                self.stop_progress()
        
        threading.Thread(target=task, daemon=True).start()