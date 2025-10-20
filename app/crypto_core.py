# app/crypto_core.py
import os
from typing import Callable, Optional
from Crypto.Cipher import Blowfish, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from .file_format import Header, ALG_BLOWFISH_CBC # <--- Membutuhkan file_format.py

BLOCK_SIZE = Blowfish.block_size  # 8

def load_rsa_public(path: str):
    return RSA.import_key(open(path, "rb").read())

def load_rsa_private(path: str, passphrase: Optional[str] = None):
    return RSA.import_key(open(path, "rb").read(), passphrase=passphrase)

def encrypt_file(
    in_path: str,
    out_path: str,
    rsa_public_path: str,
    progress_cb: Optional[Callable[[float], None]] = None
):
    # 1) kunci Blowfish acak + IV
    bf_key = get_random_bytes(32)  # 256-bit (Blowfish mendukung hingga 448-bit)
    iv = get_random_bytes(BLOCK_SIZE)

    # 2) bungkus kunci dengan RSA-OAEP
    rsa_pub = load_rsa_public(rsa_public_path)
    cipher_rsa = PKCS1_OAEP.new(rsa_pub)
    wrapped_key = cipher_rsa.encrypt(bf_key)

    # 3) tulis header
    header = Header(ALG_BLOWFISH_CBC, iv, wrapped_key)

    total = os.path.getsize(in_path)
    processed = 0

    # 4) enkripsi payload (streaming + padding di akhir)
    cipher = Blowfish.new(bf_key, Blowfish.MODE_CBC, iv=iv)

    # Buffer untuk memastikan panjang kelipatan block kecuali chunk terakhir
    buffer = b""
    chunk_size = 1024 * 1024  # 1MB

    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        fout.write(header.to_bytes())
        while True:
            chunk = fin.read(chunk_size)
            if not chunk:
                # last block -> pad and encrypt
                if buffer:
                    last = pad(buffer, BLOCK_SIZE)
                    fout.write(cipher.encrypt(last))
                    processed += len(buffer)
                break
            buffer += chunk
            # ambil bagian yang kelipatan block
            cut_len = (len(buffer) // BLOCK_SIZE) * BLOCK_SIZE
            if cut_len > 0:
                to_enc = buffer[:cut_len]
                buffer = buffer[cut_len:]
                fout.write(cipher.encrypt(to_enc))
                processed += len(to_enc)

            if progress_cb and total > 0:
                progress_cb(min(0.999, processed / total))  # sisakan untuk padding

        # jika buffer kosong tadi, tetap update progress 100% setelah padding
        if progress_cb:
            progress_cb(1.0)

def decrypt_file(
    in_path: str,
    out_path: str,
    rsa_private_path: str,
    passphrase: Optional[str] = None,
    progress_cb: Optional[Callable[[float], None]] = None
):
    with open(in_path, "rb") as fin:
        header = Header.from_file(fin)
        if header.alg != ALG_BLOWFISH_CBC:
            raise ValueError("Algoritma tidak didukung.")
        # buka private key
        rsa_priv = load_rsa_private(rsa_private_path, passphrase=passphrase)
        cipher_rsa = PKCS1_OAEP.new(rsa_priv)
        bf_key = cipher_rsa.decrypt(header.rsa_wrapped_key)

        cipher = Blowfish.new(bf_key, Blowfish.MODE_CBC, iv=header.iv)

        # ukuran payload
        fin.seek(0, os.SEEK_END)
        end = fin.tell()
        payload_start = len(header.to_bytes())
        payload_size = end - payload_start
        fin.seek(payload_start, os.SEEK_SET)

        # Dekripsi streaming, unpad di akhir
        total = payload_size
        processed = 0
        chunk_size = 1024 * 1024

        with open(out_path, "wb") as fout:
            buffer = b""
            while True:
                chunk = fin.read(chunk_size)
                if not chunk:
                    # last: decrypt + unpad
                    if buffer:
                        # Pastikan kelipatan blok untuk decrypt
                        cut_len = (len(buffer) // BLOCK_SIZE) * BLOCK_SIZE
                        last_ct = buffer[:cut_len]
                        rem = buffer[cut_len:]
                        if rem:
                            raise ValueError("Ciphertext tidak sejajar blok.")
                        last_pt = cipher.decrypt(last_ct)
                        last_pt = unpad(last_pt, BLOCK_SIZE)
                        fout.write(last_pt)
                        processed += len(last_ct)
                    break

                buffer += chunk
                # proses blok penuh kecuali simpan sisa untuk akhir
                cut_len = (len(buffer) // BLOCK_SIZE) * BLOCK_SIZE
                if cut_len > BLOCK_SIZE:  # sisakan setidaknya 1 blok untuk akhir (padding)
                    to_dec = buffer[:cut_len - BLOCK_SIZE]
                    buffer = buffer[cut_len - BLOCK_SIZE:]
                    fout.write(cipher.decrypt(to_dec))
                    processed += len(to_dec)

                if progress_cb and total > 0:
                    progress_cb(min(0.999, processed / total))

            if progress_cb:
                progress_cb(1.0)