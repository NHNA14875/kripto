# Skrip untuk membuat key baru
import os
from Crypto.PublicKey import RSA

KEY_DIR = os.path.dirname(__file__)
PRIVATE_KEY_PATH = os.path.join(KEY_DIR, "rsa_private.pem")
PUBLIC_KEY_PATH = os.path.join(KEY_DIR, "rsa_public.pem")

def generate_keys(passphrase=None):
    """Generates and saves new RSA 2048-bit keys."""
    print("Membuat RSA key pair (2048 bit)...")
    key = RSA.generate(2048)
    
    # Ekspor private key
    try:
        with open(PRIVATE_KEY_PATH, "wb") as f:
            f.write(key.export_key(format="PEM", passphrase=passphrase))
        print(f"Private key disimpan di: {PRIVATE_KEY_PATH}")

        # Ekspor public key
        with open(PUBLIC_KEY_PATH, "wb") as f:
            f.write(key.publickey().export_key(format="PEM"))
        print(f"Public key disimpan di: {PUBLIC_KEY_PATH}")
        print("Selesai.")
        
    except Exception as e:
        print(f"Error saat menyimpan key: {e}")

if __name__ == "__main__":
    # Ganti "your_passphrase" dengan password Anda, atau None jika tidak ingin pakai password
    generate_keys(passphrase=None)