# app/file_format.py
"""
Definisi Header untuk format file .venc
"""
import struct

# Tanda (magic bytes), Versi, Algoritma, IV, Panjang Key, Key
# "VENC" (4s), Versi (H=2b), Alg (H=2b), IV (8s), KeyLen (H=2b), Key (256s)
HEADER_FORMAT = "!4sHH8sH256s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAGIC = b'VENC'
VERSION = 1
ALG_BLOWFISH_CBC = 1

class Header:
    def __init__(self, alg, iv, rsa_wrapped_key):
        self.magic = MAGIC
        self.version = VERSION
        self.alg = alg
        self.iv = iv
        
        if len(rsa_wrapped_key) != 256:
            raise ValueError("RSA wrapped key harus 256 bytes (untuk RSA-2048)")
        self.rsa_wrapped_key = rsa_wrapped_key
        self.key_len = len(rsa_wrapped_key) # 256

    def to_bytes(self):
        return struct.pack(
            HEADER_FORMAT,
            self.magic,
            self.version,
            self.alg,
            self.iv,
            self.key_len,
            self.rsa_wrapped_key
        )

    @classmethod
    def from_bytes(cls, data):
        if len(data) != HEADER_SIZE:
            raise ValueError("Data header tidak valid (ukuran salah)")
        
        magic, ver, alg, iv, key_len, key = struct.unpack(HEADER_FORMAT, data)
        
        if magic != MAGIC:
            raise ValueError("File bukan format VENC (magic bytes salah)")
        if ver != VERSION:
            raise ValueError(f"Versi file tidak didukung (versi {ver})")
        if alg != ALG_BLOWFISH_CBC:
            raise ValueError(f"Algoritma file tidak didukung (alg {alg})")
        if key_len != 256:
             raise ValueError("Header korup (key_len salah)")

        return cls(alg, iv, key)

    @classmethod
    def from_file(cls, f):
        data = f.read(HEADER_SIZE)
        return cls.from_bytes(data)