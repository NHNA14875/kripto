# run.py
from app.gui import App
import sys
import os

# Menambahkan path 'app' ke sys.path jika belum ada
# (Berguna jika 'app' tidak diinstal sebagai package)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except ImportError as e:
        print("="*50)
        print("!!! IMPORT ERROR !!!")
        print(f"Gagal mengimpor: {e}")
        print("\nPastikan Anda sudah menginstal semua dependensi dari requirements.txt")
        print("Jalankan perintah ini di terminal Anda:")
        print("pip install -r requirements.txt")
        print("="*50)
        input("\nTekan Enter untuk keluar...")
    except Exception as e:
        print(f"Terjadi error saat menjalankan aplikasi: {e}")
        input("\nTekan Enter untuk keluar...")