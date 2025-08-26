const crypto = require('crypto');

const ALGO = 'aes-256-gcm';

function getKey() {
  const key = process.env.DATA_ENCRYPTION_KEY;
  if (!key) return null;
  // Accept hex or base64 or raw 32-byte string
  try {
    if (key.length === 64 && /^[0-9a-fA-F]+$/.test(key)) {
      return Buffer.from(key, 'hex');
    }
    const buf = Buffer.from(key, 'base64');
    if (buf.length === 32) return buf;
  } catch (_) {}
  // Fallback: if provided as plain string of length 32
  if (key.length === 32) return Buffer.from(key, 'utf8');
  return null;
}

function encryptJson(value) {
  try {
    const payload = typeof value === 'string' ? value : JSON.stringify(value);
    const key = getKey();
    if (!key) return payload; // no encryption configured
    const iv = crypto.randomBytes(12);
    const cipher = crypto.createCipheriv(ALGO, key, iv);
    const enc = Buffer.concat([cipher.update(payload, 'utf8'), cipher.final()]);
    const authTag = cipher.getAuthTag();
    return Buffer.concat([iv, authTag, enc]).toString('base64');
  } catch (e) {
    // On failure, return plaintext to avoid data loss
    return typeof value === 'string' ? value : JSON.stringify(value);
  }
}

function decryptJson(value) {
  try {
    const key = getKey();
    if (!key) return JSON.parse(value);
    const buf = Buffer.from(value, 'base64');
    const iv = buf.subarray(0, 12);
    const authTag = buf.subarray(12, 28);
    const enc = buf.subarray(28);
    const decipher = crypto.createDecipheriv(ALGO, key, iv);
    decipher.setAuthTag(authTag);
    const dec = Buffer.concat([decipher.update(enc), decipher.final()]).toString('utf8');
    return JSON.parse(dec);
  } catch (e) {
    // Best effort: try plain JSON
    try {
      return JSON.parse(value);
    } catch (_) {
      return null;
    }
  }
}

module.exports = { encryptJson, decryptJson };


