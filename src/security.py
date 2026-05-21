"""
Local security utilities for Mind-Shield+.

Provides lightweight on-device encryption primitives for local-only storage.
No cloud services and no telemetry.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from typing import Any, Dict


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


class LocalCipher:
    """Simple local-only stream cipher derived from an on-device key file.

    This is intentionally dependency-free so it works in the current desktop-only
    repository without external crypto packages. The key is stored locally and the
    encrypted payload is never uploaded anywhere.
    """

    def __init__(self, key_path: str | None = None):
        self.key_path = key_path or os.path.join(DATA_DIR, ".mindshield.key")
        self._master_key = self._load_or_create_key()

    def _load_or_create_key(self) -> bytes:
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as f:
                return f.read()
        key = secrets.token_bytes(32)
        with open(self.key_path, "wb") as f:
            f.write(key)
        return key

    def _keystream(self, nonce: bytes, length: int) -> bytes:
        out = bytearray()
        counter = 0
        while len(out) < length:
            block = hmac.new(
                self._master_key,
                nonce + counter.to_bytes(8, "big"),
                hashlib.sha256,
            ).digest()
            out.extend(block)
            counter += 1
        return bytes(out[:length])

    def encrypt_bytes(self, data: bytes) -> bytes:
        nonce = secrets.token_bytes(16)
        stream = self._keystream(nonce, len(data))
        cipher = bytes(a ^ b for a, b in zip(data, stream))
        return base64.urlsafe_b64encode(nonce + cipher)

    def decrypt_bytes(self, token: bytes) -> bytes:
        raw = base64.urlsafe_b64decode(token)
        nonce, cipher = raw[:16], raw[16:]
        stream = self._keystream(nonce, len(cipher))
        return bytes(a ^ b for a, b in zip(cipher, stream))

    def encrypt_text(self, text: str) -> str:
        return self.encrypt_bytes(text.encode("utf-8")).decode("utf-8")

    def decrypt_text(self, text: str) -> str:
        return self.decrypt_bytes(text.encode("utf-8")).decode("utf-8")

    def encrypt_json(self, payload: Dict[str, Any]) -> str:
        return self.encrypt_text(json.dumps(payload, ensure_ascii=False))

    def decrypt_json(self, payload: str) -> Dict[str, Any]:
        return json.loads(self.decrypt_text(payload))


@dataclass
class SecureRecord:
    """Convenience wrapper for encrypted local payloads."""

    data: Dict[str, Any]

    def to_encrypted_text(self, cipher: LocalCipher) -> str:
        return cipher.encrypt_json(self.data)

    @staticmethod
    def from_encrypted_text(cipher: LocalCipher, text: str) -> "SecureRecord":
        return SecureRecord(cipher.decrypt_json(text))
