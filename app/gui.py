# app/gui.py - FIXED VERSION
import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import threading
import traceback
import os
import time
import datetime
from pathlib import Path
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .crypto_core import encrypt_file, decrypt_file
from .metrics_core import compare_videos_advanced, no_reference_metrics
from .utils import save_metrics_summary, file_byte_histogram, compute_entropy_from_hist

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class VideoPlayer:
    def __init__(self, canvas_widget, video_path):
        self.canvas = canvas_widget
        self.video_path = video_path
        self.cap = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.delay = 33

        if video_path and os.path.exists(video_path):
            self.cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.delay = max(1, int(1000 / self.fps))

    def play(self):
        if not self.cap or not self.cap.isOpened():
            return
        self.is_playing = True
        self._play_frame()

    def pause(self):
        self.is_playing = False

    def stop(self):
        self.is_playing = False
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self._show_current_frame()

    def _play_frame(self):
        if not self.is_playing or not self.cap:
            return

        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._display_frame(frame)
            self.canvas.after(self.delay, self._play_frame)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self._play_frame()

    def _show_current_frame(self):
        if not self.cap:
            return
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos < 0:
            pos = 0
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._display_frame(frame)

    def _display_frame(self, frame):
        try:
            self.canvas.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width <= 1:
                canvas_width = 400
            if canvas_height <= 1:
                canvas_height = 300

            h, w = frame.shape[:2]
            aspect = w / h

            if canvas_width / canvas_height > aspect:
                new_height = canvas_height - 20
                new_width = int(new_height * aspect)
            else:
                new_width = canvas_width - 20
                new_height = int(new_width / aspect)

            frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            img = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(img)

            self.canvas.configure(image=photo, text="")
            self.canvas.image = photo
        except Exception as e:
            print(f"Error displaying frame: {e}")

    def release(self):
        self.is_playing = False
        if self.cap:
            self.cap.release()


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🔐 Video Crypto Modern - Blowfish + RSA")
        self.geometry("1450x850")
        self.minsize(1200, 650)

        base_path = Path(__file__).resolve().parent.parent
        self.configure(fg_color='#0f172a')

        self.colors = {
            'primary': '#3b82f6',
            'primary_hover': '#2563eb',
            'success': '#10b981',
            'success_hover': '#059669',
            'danger': '#ef4444',
            'dark_bg': '#1e293b',
            'card_bg': '#334155',
            'text': '#f1f5f9',
            'text_secondary': '#94a3b8'
        }

        self.input_path = ctk.StringVar()
        self.output_path = ctk.StringVar()
        self.pubkey_path = ctk.StringVar(value=str(base_path / "keys" / "rsa_public.pem"))
        self.privkey_path = ctk.StringVar(value=str(base_path / "keys" / "rsa_private.pem"))
        self.is_running = False

        self.histogram_data = {'orig': None, 'enc': None, 'dec': None}
        self.player_orig = None
        self.player_dec = None
        self.video_paths = {'orig': None, 'dec': None}

        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=5)
        self.grid_rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_right_panel()

        self._show_video_placeholder()
        self._show_histogram_placeholder()

        self.bind("<Control-e>", lambda e: self.on_encrypt())
        self.bind("<Control-d>", lambda e: self.on_decrypt())
        self.bind("<Control-m>", lambda e: self.on_advanced_test())

    def _build_left_panel(self):
        left_frame = ctk.CTkFrame(self, corner_radius=15, fg_color='#1e293b', border_width=1, border_color='#334155')
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(15, 8), pady=15)
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(5, weight=1)

        header = ctk.CTkFrame(left_frame, fg_color='#334155', corner_radius=12, height=60)
        header.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 10))
        header.grid_propagate(False)

        header_content = ctk.CTkFrame(header, fg_color="transparent")
        header_content.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(header_content, text="🔐", font=("Segoe UI", 28)).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(header_content, text="Kontrol Enkripsi/Dekripsi",
                     font=("Segoe UI", 20, "bold"), text_color='#f1f5f9').pack(side="left")

        self._build_path_frame(left_frame)
        self._build_action_frame(left_frame)

        progress_container = ctk.CTkFrame(left_frame, fg_color='transparent')
        progress_container.grid(row=3, column=0, sticky="ew", padx=15, pady=(10, 10))

        self.progress = ctk.CTkProgressBar(progress_container, mode="determinate", corner_radius=8, height=8,
                                           progress_color='#3b82f6')
        self.progress.set(0)
        self.progress.pack(fill="x")

        self._build_table_section(left_frame)

    def _build_path_frame(self, parent):
        frame = ctk.CTkFrame(parent, corner_radius=12, fg_color='#334155', border_width=1, border_color='#475569')
        frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=8)
        frame.grid_columnconfigure(1, weight=1)

        label_font = ("Segoe UI", 11, "bold")
        entry_style = {'corner_radius': 8, 'border_width': 1, 'border_color': '#64748b',
                       'fg_color': '#1e293b', 'font': ("Segoe UI", 10)}
        btn_style = {'corner_radius': 8, 'width': 90, 'height': 32, 'font': ("Segoe UI", 10, "bold"),
                     'fg_color': self.colors['primary'], 'hover_color': self.colors['primary_hover']}

        ctk.CTkLabel(frame, text="📁 Input (Video/VENC):", font=label_font, text_color='#cbd5e1').grid(
            row=0, column=0, padx=15, pady=(15, 8), sticky="w")
        e1 = ctk.CTkEntry(frame, textvariable=self.input_path, **entry_style)
        e1.grid(row=0, column=1, padx=8, pady=(15, 8), sticky="ew")
        ctk.CTkButton(frame, text="Browse", command=self.browse_input, **btn_style).grid(
            row=0, column=2, padx=(8, 15), pady=(15, 8))

        ctk.CTkLabel(frame, text="💾 Output (VENC/Video):", font=label_font, text_color='#cbd5e1').grid(
            row=1, column=0, padx=15, pady=8, sticky="w")
        e2 = ctk.CTkEntry(frame, textvariable=self.output_path, **entry_style)
        e2.grid(row=1, column=1, padx=8, pady=8, sticky="ew")
        ctk.CTkButton(frame, text="Browse", command=self.browse_output, **btn_style).grid(
            row=1, column=2, padx=(8, 15), pady=8)

        ctk.CTkLabel(frame, text="🔑 Public Key:", font=label_font, text_color='#cbd5e1').grid(
            row=2, column=0, padx=15, pady=8, sticky="w")
        e3 = ctk.CTkEntry(frame, textvariable=self.pubkey_path, **entry_style)
        e3.grid(row=2, column=1, padx=8, pady=8, sticky="ew")
        ctk.CTkButton(frame, text="Browse", command=lambda: self.browse_key(self.pubkey_path, "Public Key"),
                      **btn_style).grid(row=2, column=2, padx=(8, 15), pady=8)

        ctk.CTkLabel(frame, text="🔐 Private Key:", font=label_font, text_color='#cbd5e1').grid(
            row=3, column=0, padx=15, pady=(8, 15), sticky="w")
        e4 = ctk.CTkEntry(frame, textvariable=self.privkey_path, **entry_style)
        e4.grid(row=3, column=1, padx=8, pady=(8, 15), sticky="ew")
        ctk.CTkButton(frame, text="Browse", command=lambda: self.browse_key(self.privkey_path, "Private Key"),
                      **btn_style).grid(row=3, column=2, padx=(8, 15), pady=(8, 15))

    def _build_action_frame(self, parent):
        frame = ctk.CTkFrame(parent, fg_color='transparent')
        frame.grid(row=2, column=0, sticky="nsew", padx=15, pady=8)
        frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.encrypt_btn = ctk.CTkButton(
            frame, text="🔒 Enkripsi", height=42, corner_radius=10, font=("Segoe UI", 13, "bold"),
            fg_color='#3b82f6', hover_color='#2563eb', command=self.on_encrypt)
        self.encrypt_btn.grid(row=0, column=0, padx=5, pady=10, sticky="ew")

        self.decrypt_btn = ctk.CTkButton(
            frame, text="🔓 Dekripsi", height=42, corner_radius=10, font=("Segoe UI", 13, "bold"),
            fg_color='#8b5cf6', hover_color='#7c3aed', command=self.on_decrypt)
        self.decrypt_btn.grid(row=0, column=1, padx=5, pady=10, sticky="ew")

        self.metrics_btn = ctk.CTkButton(
            frame, text="📊 Uji Lanjutan", height=42, corner_radius=10, font=("Segoe UI", 13, "bold"),
            fg_color='#10b981', hover_color='#059669', command=self.on_advanced_test)
        self.metrics_btn.grid(row=0, column=2, padx=5, pady=10, sticky="ew")

    def _build_table_section(self, parent):
        table_container = ctk.CTkFrame(parent, corner_radius=12, fg_color='#334155',
                                       border_width=1, border_color='#475569')
        table_container.grid(row=4, column=0, rowspan=2, sticky="nsew", padx=15, pady=(8, 15))
        table_container.grid_rowconfigure(1, weight=1)
        table_container.grid_columnconfigure(0, weight=1)

        table_header = ctk.CTkFrame(table_container, fg_color='#475569', corner_radius=10, height=50)
        table_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        table_header.grid_propagate(False)
        table_header.grid_columnconfigure(0, weight=1)

        header_left = ctk.CTkFrame(table_header, fg_color="transparent")
        header_left.place(relx=0.02, rely=0.5, anchor="w")

        ctk.CTkLabel(header_left, text="📊", font=("Segoe UI", 18)).pack(side="left", padx=(5, 8))
        ctk.CTkLabel(header_left, text="Hasil Metrik (Tabel)",
                     font=("Segoe UI", 15, "bold"), text_color='#f1f5f9').pack(side="left")

        clear_btn = ctk.CTkButton(table_header, text="🗑️ Clear", width=90, height=32, corner_radius=8,
                                  font=("Segoe UI", 10, "bold"), fg_color='#ef4444', hover_color='#dc2626',
                                  command=self.clear_table)
        clear_btn.place(relx=0.98, rely=0.5, anchor="e")

        table_frame = ctk.CTkFrame(table_container, fg_color='#1e293b')
        table_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        cols = ["Jenis", "PSNR", "SSIM", "LPIPS", "MUSIQ", "NRQM", "NIQE", "BRISQUE",
                "Sharpness", "FileEntropy", "UkuranKB", "CompressionRatio", "Timestamp"]
        self.metrics_table = ttk.Treeview(table_frame, columns=cols, show="headings", height=6)

        width_map = {
            "Jenis": 110, "PSNR": 95, "SSIM": 95, "LPIPS": 95, "MUSIQ": 95, "NRQM": 95,
            "NIQE": 95, "BRISQUE": 95, "Sharpness": 110, "FileEntropy": 120,
            "UkuranKB": 95, "CompressionRatio": 130, "Timestamp": 150
        }

        for col in cols:
            self.metrics_table.heading(col, text=col)
            self.metrics_table.column(col, anchor="center", width=width_map.get(col, 95), stretch=False)

        self.metrics_table.grid(row=0, column=0, sticky="nsew")

        scroll_y = ttk.Scrollbar(table_frame, orient="vertical", command=self.metrics_table.yview)
        self.metrics_table.configure(yscrollcommand=scroll_y.set)
        scroll_y.grid(row=0, column=1, sticky="ns")

        scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.metrics_table.xview)
        self.metrics_table.configure(xscrollcommand=scroll_x.set)
        scroll_x.grid(row=1, column=0, sticky="ew")

        style = ttk.Style()
        style.theme_use('default')
        style.configure("Treeview", background="#1e293b", foreground="#e2e8f0",
                        fieldbackground="#1e293b", borderwidth=0, rowheight=30, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background="#334155", foreground="#f1f5f9",
                        borderwidth=0, font=("Segoe UI", 10, "bold"))
        style.map('Treeview', background=[('selected', '#3b82f6')], foreground=[('selected', '#ffffff')])

    def clear_table(self):
        for i in self.metrics_table.get_children():
            self.metrics_table.delete(i)

    def compute_encrypted_metrics(self, enc_path):
        if not enc_path or not os.path.exists(enc_path):
            return {}
        try:
            with open(enc_path, 'rb') as f:
                data = f.read(100 * 1024)

            byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
            expected = len(data) / 256 if len(data) > 0 else 1
            chi_square = float(np.sum((byte_counts - expected) ** 2 / expected))

            if len(data) > 1:
                arr = np.frombuffer(data, dtype=np.uint8)
                correlation = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
            else:
                correlation = 0.0

            std_dev = float(np.std(byte_counts))
            entropy = self.histogram_data.get('enc', {}).get('entropy', 0.0)
            quality = min(100.0, (entropy / 8.0) * 100.0 * (1.0 - abs(correlation)))

            return {
                'CHI_SQUARE': f"{chi_square:.2f}",
                'CORRELATION': f"{correlation:.6f}",
                'FREQ_STD': f"{std_dev:.2f}",
                'RANDOM_QUALITY': f"{quality:.2f}%"
            }
        except Exception:
            return {}

    def table_add_rows(self, ref, noref_orig, noref_dec, paths, enc_metrics=None):
        self.clear_table()
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        def kb(path):
            try:
                return f"{os.path.getsize(path) / 1024:.2f}"
            except Exception:
                return "-"

        def file_size(path):
            try:
                return os.path.getsize(path)
            except Exception:
                return 0

        def safe_fmt(val):
            """Format nilai dengan aman"""
            if val is None or val == '-':
                return '-'
            if isinstance(val, (int, float)):
                return f"{val:.2f}"
            return str(val)

        ent_orig = self.histogram_data.get('orig', {}).get('entropy')
        ent_enc = self.histogram_data.get('enc', {}).get('entropy')
        ent_dec = self.histogram_data.get('dec', {}).get('entropy')
        ent_fmt = lambda v: f"{v:.6f}" if isinstance(v, (int, float)) else "-"

        size_orig = file_size(paths.get("orig"))
        size_enc = file_size(paths.get("enc")) if paths.get("enc") else 0
        size_dec = file_size(paths.get("dec"))

        ratio_enc = f"{(size_enc / size_orig):.4f}" if size_orig > 0 and size_enc > 0 else "-"
        ratio_dec = f"{(size_dec / size_orig):.4f}" if size_orig > 0 and size_dec > 0 else "-"

        # Row: Asli
        row_orig = (
            "Asli", "-", "-", "-",
            safe_fmt(noref_orig.get('MUSIQ')),
            safe_fmt(noref_orig.get('NRQM')),
            safe_fmt(noref_orig.get('NIQE')),
            safe_fmt(noref_orig.get('BRISQUE')),
            safe_fmt(noref_orig.get('SHARPNESS')),
            ent_fmt(ent_orig), kb(paths.get("orig")), "1.0000", now
        )
        self.metrics_table.insert("", "end", values=row_orig)

        # Row: Terenkripsi
        enc_display = {}
        if enc_metrics:
            enc_display = {
                'PSNR': f"χ²={enc_metrics.get('CHI_SQUARE', '-')}",
                'SSIM': f"r={enc_metrics.get('CORRELATION', '-')}",
                'LPIPS': f"σ={enc_metrics.get('FREQ_STD', '-')}",
                'RANDOM': enc_metrics.get('RANDOM_QUALITY', '-')
            }

        row_enc = (
            "Terenkripsi",
            enc_display.get('PSNR', 'N/A'), enc_display.get('SSIM', 'N/A'), enc_display.get('LPIPS', 'N/A'),
            enc_display.get('RANDOM', 'N/A'), "N/A", "N/A", "N/A", "N/A",
            ent_fmt(ent_enc), kb(paths.get("enc")) if paths.get("enc") else "-", ratio_enc, now
        )
        self.metrics_table.insert("", "end", values=row_enc)

        # Row: Didekripsi
        psnr = ref.get('PSNR', '-')
        ssim = ref.get('SSIM', '-')
        lpips_val = ref.get('LPIPS', '-')

        if isinstance(psnr, (int, float)):
            psnr = "100.00" if psnr >= 100 else f"{psnr:.2f}"
        if isinstance(ssim, (int, float)):
            ssim = f"{ssim:.6f}"
        if isinstance(lpips_val, (int, float)):
            lpips_val = f"{lpips_val:.6f}"

        row_dec = (
            "Didekripsi", psnr, ssim, lpips_val,
            safe_fmt(noref_dec.get('MUSIQ')),
            safe_fmt(noref_dec.get('NRQM')),
            safe_fmt(noref_dec.get('NIQE')),
            safe_fmt(noref_dec.get('BRISQUE')),
            safe_fmt(noref_dec.get('SHARPNESS')),
            ent_fmt(ent_dec), kb(paths.get("dec")), ratio_dec, now
        )
        self.metrics_table.insert("", "end", values=row_dec)

    def _build_right_panel(self):
        right_frame = ctk.CTkFrame(self, corner_radius=15, fg_color='#1e293b', border_width=1, border_color='#334155')
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 15), pady=15)
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_rowconfigure(1, weight=1)

        self._build_video_player_section(right_frame)
        self._build_histogram_section(right_frame)

    def _build_video_player_section(self, parent):
        video_container = ctk.CTkFrame(parent, fg_color='transparent')
        video_container.grid(row=0, column=0, sticky="nsew", padx=15, pady=(15, 8))
        video_container.grid_columnconfigure((0, 1), weight=1)
        video_container.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(video_container, fg_color='#334155', corner_radius=10, height=50)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=(0, 10))
        header.grid_propagate(False)

        header_content = ctk.CTkFrame(header, fg_color="transparent")
        header_content.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(header_content, text="🎬", font=("Segoe UI", 22)).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(header_content, text="Video Player", font=("Segoe UI", 17, "bold"),
                     text_color='#f1f5f9').pack(side="left", padx=(0, 15))
        ctk.CTkLabel(header_content, text="Klik Play untuk memutar", font=("Segoe UI", 10),
                     text_color='#94a3b8').pack(side="left")

        self._build_single_player(video_container, 0, "Video Asli", "orig")
        self._build_single_player(video_container, 1, "Video Hasil Dekripsi", "dec")

    def _build_single_player(self, parent, col, title, player_type):
        frame = ctk.CTkFrame(parent, corner_radius=12, fg_color='#334155', border_width=1, border_color='#475569')
        frame.grid(row=1, column=col, sticky="nsew", padx=(0 if col == 0 else 8, 8 if col == 0 else 0), pady=(0, 0))
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        title_frame = ctk.CTkFrame(frame, fg_color='#475569', corner_radius=10, height=45)
        title_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        title_frame.grid_propagate(False)

        ctk.CTkLabel(title_frame, text=title, font=("Segoe UI", 14, "bold"),
                     text_color='#f1f5f9').place(relx=0.5, rely=0.5, anchor="center")

        canvas_container = ctk.CTkFrame(frame, fg_color='#1e293b', corner_radius=10)
        canvas_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)

        canvas = ctk.CTkLabel(canvas_container, text="", fg_color='#1e293b')
        canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        if player_type == "orig":
            self.video_orig_canvas = canvas
        else:
            self.video_dec_canvas = canvas

        control_frame = ctk.CTkFrame(frame, fg_color='transparent')
        control_frame.grid(row=2, column=0, pady=(0, 12))

        btn_style = {'width': 95, 'height': 36, 'corner_radius': 8, 'font': ("Segoe UI", 11, "bold")}

        ctk.CTkButton(control_frame, text="▶ Play", fg_color='#10b981', hover_color='#059669',
                      command=lambda: self._play_video(player_type), **btn_style).pack(side="left", padx=4)
        ctk.CTkButton(control_frame, text="⏸ Pause", fg_color='#f59e0b', hover_color='#d97706',
                      command=lambda: self._pause_video(player_type), **btn_style).pack(side="left", padx=4)
        ctk.CTkButton(control_frame, text="⏹ Stop", fg_color='#ef4444', hover_color='#dc2626',
                      command=lambda: self._stop_video(player_type), **btn_style).pack(side="left", padx=4)

    def _play_video(self, player_type):
        player = self.player_orig if player_type == "orig" else self.player_dec
        if player:
            player.play()

    def _pause_video(self, player_type):
        player = self.player_orig if player_type == "orig" else self.player_dec
        if player:
            player.pause()

    def _stop_video(self, player_type):
        player = self.player_orig if player_type == "orig" else self.player_dec
        if player:
            player.stop()

    def _build_histogram_section(self, parent):
        hist_container = ctk.CTkFrame(parent, fg_color='transparent')
        hist_container.grid(row=1, column=0, sticky="nsew", padx=15, pady=(8, 15))
        hist_container.grid_columnconfigure(0, weight=1)
        hist_container.grid_rowconfigure(1, weight=1)

        hist_header = ctk.CTkFrame(hist_container, fg_color='#334155', corner_radius=10, height=50)
        hist_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        hist_header.grid_propagate(False)

        header_content = ctk.CTkFrame(hist_header, fg_color="transparent")
        header_content.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(header_content, text="📈", font=("Segoe UI", 22)).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(header_content, text="Histogram Distribusi Byte", font=("Segoe UI", 17, "bold"),
                     text_color='#f1f5f9').pack(side="left", padx=(0, 15))
        ctk.CTkLabel(header_content, text="Analisis entropi file", font=("Segoe UI", 10),
                     text_color='#94a3b8').pack(side="left")

        self.histogram_frame = ctk.CTkFrame(hist_container, corner_radius=12, fg_color='#334155',
                                            border_width=1, border_color='#475569')
        self.histogram_frame.grid(row=1, column=0, sticky="nsew")
        self.histogram_frame.grid_columnconfigure(0, weight=1)
        self.histogram_frame.grid_rowconfigure(0, weight=1)

    def _show_video_placeholder(self):
        if hasattr(self, "video_orig_canvas"):
            self.video_orig_canvas.configure(
                text="Tidak ada video\n\nJalankan 'Uji Lanjutan'\nuntuk load video",
                font=("Segoe UI", 12), text_color='#64748b'
            )
        if hasattr(self, "video_dec_canvas"):
            self.video_dec_canvas.configure(
                text="Tidak ada video\n\nJalankan 'Uji Lanjutan'\nuntuk load video",
                font=("Segoe UI", 12), text_color='#64748b'
            )

    def _show_histogram_placeholder(self):
        for widget in self.histogram_frame.winfo_children():
            widget.destroy()
        ctk.CTkLabel(
            self.histogram_frame,
            text="Tidak ada histogram\n\nJalankan 'Uji Lanjutan' untuk melihat hasil",
            font=("Segoe UI", 14), text_color='#64748b'
        ).place(relx=0.5, rely=0.5, anchor="center")

    def update_video_players(self):
        try:
            if self.player_orig:
                self.player_orig.release()
            if self.player_dec:
                self.player_dec.release()

            if self.video_paths['orig']:
                self.player_orig = VideoPlayer(self.video_orig_canvas, self.video_paths['orig'])
                self.player_orig._show_current_frame()

            if self.video_paths['dec']:
                self.player_dec = VideoPlayer(self.video_dec_canvas, self.video_paths['dec'])
                self.player_dec._show_current_frame()
        except Exception:
            pass

    def update_histogram_display(self):
        for widget in self.histogram_frame.winfo_children():
            widget.destroy()

        if not any(self.histogram_data.values()):
            self._show_histogram_placeholder()
            return

        try:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
            fig.patch.set_facecolor('#1e293b')

            titles = ['Video Asli', 'File Terenkripsi', 'Video Dekripsi']
            colors = ['#10b981', '#ef4444', '#3b82f6']
            keys = ['orig', 'enc', 'dec']

            for ax, title, color, key in zip(axes, titles, colors, keys):
                data = self.histogram_data.get(key)
                ax.set_facecolor('#0f172a')

                if data and data.get('hist') is not None:
                    hist = data['hist']
                    entropy = data.get('entropy', 0)
                    x = np.arange(256)

                    ax.bar(x, hist, color=color, alpha=0.85, width=1.0, edgecolor=color, linewidth=0.5)
                    ax.set_title(f'{title}\nEntropy: {entropy:.6f}', color='#f1f5f9',
                                 fontweight='bold', fontsize=12, pad=10)
                    ax.set_xlabel('Byte Value', color='#cbd5e1', fontsize=10, labelpad=8)
                    ax.set_ylabel('Frequency', color='#cbd5e1', fontsize=10, labelpad=8)
                    ax.tick_params(colors='#94a3b8', labelsize=8)
                    ax.grid(True, alpha=0.15, color='#475569', linestyle='--', linewidth=0.8)
                    ax.set_xlim(0, 255)
                    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

                    for spine in ax.spines.values():
                        spine.set_edgecolor('#334155')
                        spine.set_linewidth(1.5)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', color='#64748b', fontsize=14,
                            transform=ax.transAxes, fontweight='bold')
                    ax.set_title(title, color='#f1f5f9', fontweight='bold', fontsize=12)
                    ax.tick_params(colors='#94a3b8')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#334155')

            plt.tight_layout(pad=2.0)
            canvas = FigureCanvasTkAgg(fig, master=self.histogram_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill="both", expand=True, padx=10, pady=10)
            self.histogram_canvas = canvas
        except Exception:
            self._show_histogram_placeholder()

    def compute_file_histogram(self, file_path, file_type):
        try:
            if not file_path or not os.path.exists(file_path):
                return
            hist = file_byte_histogram(file_path)
            entropy = compute_entropy_from_hist(hist)
            self.histogram_data[file_type] = {'hist': hist, 'entropy': entropy, 'path': file_path}
        except Exception:
            pass

    def log(self, message):
        return

    def update_status(self, message):
        return

    def set_progress(self, value):
        self.after(0, lambda: self.progress.set(value))

    def start_indeterminate(self):
        self.after(0, lambda: (self.progress.configure(mode="indeterminate"), self.progress.start()))

    def stop_progress(self):
        self.after(0, lambda: (self.progress.stop(), self.progress.configure(mode="determinate"), self.progress.set(0)))

    def set_running(self, is_running):
        self.is_running = is_running
        state = "disabled" if is_running else "normal"

        def _apply():
            self.encrypt_btn.configure(state=state)
            self.decrypt_btn.configure(state=state)
            self.metrics_btn.configure(state=state)

        self.after(0, _apply)

    def browse_input(self):
        p = filedialog.askopenfilename(
            title="Pilih file input",
            filetypes=[("Video/Encrypted", "*.mp4;*.mkv;*.avi;*.venc"), ("All", "*.*")]
        )
        if not p:
            return
        self.input_path.set(p)
        if p.lower().endswith(".venc"):
            self.output_path.set(os.path.splitext(p)[0] + "_DECRYPTED.mp4")
        else:
            self.output_path.set(p + ".venc")

    def browse_output(self):
        init_name = ""
        inp = self.input_path.get()
        if inp:
            if inp.lower().endswith(".venc"):
                init_name = os.path.splitext(inp)[0] + "_DECRYPTED.mp4"
            else:
                init_name = inp + ".venc"
        p = filedialog.asksaveasfilename(title="Simpan sebagai", initialfile=os.path.basename(init_name) if init_name else None)
        if p:
            self.output_path.set(p)

    def browse_key(self, str_var, title):
        p = filedialog.askopenfilename(title=f"Pilih {title}", filetypes=[("PEM Files", "*.pem")])
        if p:
            str_var.set(p)

    def run_task(self, task_function, *args):
        if self.is_running:
            messagebox.showwarning("Sedang Berjalan", "Operasi lain sedang berjalan.")
            return

        def worker():
            self.set_running(True)
            try:
                task_function(*args)
            except Exception as e:
                messagebox.showerror("Error", f"Terjadi error: {e}\n\n{traceback.format_exc()}")
            finally:
                self.set_running(False)
                self.stop_progress()
                self.update_status("Siap")

        threading.Thread(target=worker, daemon=True).start()

    def on_encrypt(self):
        in_p = self.input_path.get()
        out_p = self.output_path.get()
        pub = self.pubkey_path.get()
        if not all([in_p, out_p, pub]):
            messagebox.showwarning("Input Kurang", "Harap isi path Input, Output, dan Public Key.")
            return

        def task():
            self.start_indeterminate()
            start_time = time.time()
            encrypt_file(in_p, out_p, pub, progress_cb=None)
            duration = time.time() - start_time
            self.stop_progress()
            messagebox.showinfo("Selesai", f"Enkripsi selesai!\nOutput: {os.path.basename(out_p)}\nDurasi: {duration:.2f} detik")

        self.run_task(task)

    def on_decrypt(self):
        in_p = self.input_path.get()
        out_p = self.output_path.get()
        priv = self.privkey_path.get()
        if not all([in_p, out_p, priv]):
            messagebox.showwarning("Input Kurang", "Harap isi path Input, Output, dan Private Key.")
            return

        def task():
            self.start_indeterminate()
            start_time = time.time()
            decrypt_file(in_p, out_p, priv, progress_cb=None)
            duration = time.time() - start_time
            self.stop_progress()
            messagebox.showinfo("Selesai", f"Dekripsi selesai!\nOutput: {os.path.basename(out_p)}\nDurasi: {duration:.2f} detik")

        self.run_task(task)

    def on_advanced_test(self):
        self.clear_table()

        orig = filedialog.askopenfilename(title="Pilih Video ASLI", filetypes=[("Video", "*.mp4;*.mkv;*.avi")])
        if not orig:
            messagebox.showwarning("Batal", "Uji dibatalkan (video asli tidak dipilih).")
            return

        enc = filedialog.askopenfilename(
            title="Pilih File TERENKRIPSI (Opsional)",
            filetypes=[("Encrypted", "*.venc"), ("All", "*.*")]
        )
        dec = filedialog.askopenfilename(title="Pilih Video DEKRIPSI", filetypes=[("Video", "*.mp4;*.mkv;*.avi")])
        if not dec:
            messagebox.showwarning("Batal", "Uji dibatalkan (video dekripsi tidak dipilih).")
            return

        self.video_paths['orig'] = orig
        self.video_paths['dec'] = dec
        paths = {"orig": orig, "enc": enc if enc else None, "dec": dec}
        out_folder = os.path.dirname(orig)

        def task():
            start_time = time.time()
            self.update_status("Menghitung metrik...")

            print("\n" + "="*80)
            print("STARTING ADVANCED METRICS TEST")
            print("="*80)

            self.set_progress(0.05)
            self.after(0, self.update_video_players)

            self.set_progress(0.15)
            print("\n[Step 1] Computing histograms...")
            self.compute_file_histogram(orig, 'orig')
            if enc:
                self.compute_file_histogram(enc, 'enc')
            self.compute_file_histogram(dec, 'dec')
            self.after(0, self.update_histogram_display)
            print("[Step 1] Histograms complete")

            enc_metrics = None
            if enc:
                self.set_progress(0.25)
                print("\n[Step 2] Computing encrypted file metrics...")
                enc_metrics = self.compute_encrypted_metrics(enc)
                print(f"[Step 2] Encrypted metrics: {enc_metrics}")

            self.set_progress(0.4)
            print("\n[Step 3] Computing reference metrics (PSNR, SSIM, LPIPS)...")
            try:
                ref = compare_videos_advanced(orig, dec)
                print(f"[Step 3] Reference metrics complete: {ref}")
            except Exception as e:
                print(f"[Step 3] ERROR: {e}")
                traceback.print_exc()
                ref = {"PSNR": "-", "SSIM": "-", "LPIPS": "-"}

            self.set_progress(0.6)
            print("\n[Step 4] Computing no-reference metrics for ORIGINAL video...")
            try:
                noref_orig = no_reference_metrics(orig)
                print(f"[Step 4] Original metrics: {noref_orig}")
            except Exception as e:
                print(f"[Step 4] ERROR: {e}")
                traceback.print_exc()
                noref_orig = {k: "-" for k in ['SHARPNESS', 'ENTROPY', 'BRISQUE', 'NIQE', 'MUSIQ', 'NRQM']}

            self.set_progress(0.8)
            print("\n[Step 5] Computing no-reference metrics for DECRYPTED video...")
            try:
                noref_dec = no_reference_metrics(dec)
                print(f"[Step 5] Decrypted metrics: {noref_dec}")
            except Exception as e:
                print(f"[Step 5] ERROR: {e}")
                traceback.print_exc()
                noref_dec = {k: "-" for k in ['SHARPNESS', 'ENTROPY', 'BRISQUE', 'NIQE', 'MUSIQ', 'NRQM']}

            print("\n[Step 6] Populating table...")
            self.table_add_rows(ref, noref_orig, noref_dec, paths, enc_metrics)

            self.set_progress(0.95)
            print("\n[Step 7] Saving summary report...")
            try:
                save_metrics_summary(ref, noref_orig, noref_dec, paths, out_folder)
                print("[Step 7] Summary saved successfully")
            except Exception as e:
                print(f"[Step 7] ERROR: {e}")
                traceback.print_exc()

            duration = time.time() - start_time
            self.stop_progress()
            
            print("\n" + "="*80)
            print(f"METRICS TEST COMPLETE - Duration: {duration:.2f}s")
            print("="*80 + "\n")
            
            messagebox.showinfo(
                "Selesai",
                "Uji metrik selesai!\n\n"
                "✓ Video player ready (klik Play untuk memutar)\n"
                "✓ Histogram ditampilkan\n"
                f"✓ Laporan disimpan di: {out_folder}\n\n"
                f"Durasi: {duration:.2f} detik"
            )

        self.run_task(task)

    def __del__(self):
        try:
            if hasattr(self, 'player_orig') and self.player_orig:
                self.player_orig.release()
            if hasattr(self, 'player_dec') and self.player_dec:
                self.player_dec.release()
        except Exception:
            pass


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()