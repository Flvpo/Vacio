from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256
from Crypto.Random import get_random_bytes
from typing import Tuple, Optional

def create_encryption_key(voice_key: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """
    Create AES encryption key from voice characteristics.
    
    Args:
        voice_key: String representation of voice characteristics
        salt: Optional salt for key derivation
        
    Returns:
        Tuple containing the encryption key and salt used
    """
    if not isinstance(voice_key, (str, bytes)) or not voice_key:
        raise ValueError("Invalid voice key")
    
    if salt is None:
        salt = get_random_bytes(16)
    elif not isinstance(salt, bytes) or len(salt) != 16:
        raise ValueError("Invalid salt")
    
    key = PBKDF2(
        password=voice_key.encode() if isinstance(voice_key, str) else voice_key,
        salt=salt,
        dkLen=32,  # AES-256
        count=200000,  # Increased iterations for security
        hmac_hash_module=SHA256
    )
    return key, salt

def encrypt_file(file_data: bytes, key: bytes) -> bytes:
    """
    Encrypt file data using AES in GCM mode.
    
    Args:
        file_data: Raw file data to encrypt
        key: Encryption key
        
    Returns:
        Encrypted data including nonce and tag
    """
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(file_data)
    
    # Combine nonce, tag, and ciphertext
    return cipher.nonce + tag + ciphertext

def decrypt_file(encrypted_data: bytes, key: bytes) -> bytes:
    """
    Decrypt file data using AES in GCM mode.
    
    Args:
        encrypted_data: Combined nonce, tag and encrypted data
        key: Decryption key
        
    Returns:
        Decrypted file data
    """
    if len(encrypted_data) < 48:  # Minimum size for nonce(16) + tag(16) + data(min 16)
        raise ValueError("Corrupted encrypted data")
        
    nonce = encrypted_data[:16]
    tag = encrypted_data[16:32]
    ciphertext = encrypted_data[32:]
    
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    data = cipher.decrypt_and_verify(ciphertext, tag)
    return data

def verify_key_format(key: bytes) -> bool:
    """
    Verify that a key meets the required format and length.
    
    Args:
        key: Key to verify
        
    Returns:
        Boolean indicating if key is valid
    """
    return isinstance(key, bytes) and len(key) == 32

def generate_random_key() -> bytes:
    """
    Generate a random encryption key.
    
    Returns:
        Random 32-byte key
    """
    return get_random_bytes(32)
        