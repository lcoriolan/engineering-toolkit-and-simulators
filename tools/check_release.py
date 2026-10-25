#!/usr/bin/env python3
"""Verify the explicit public allowlist, content hashes, and common leak patterns.

--refresh updates hashes only for already allowlisted, reviewed files. New files
must be added deliberately to PUBLIC_FILES.json; they are never auto-included.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

PATTERNS = {
    'personal absolute path': rb'(?<![A-Za-z0-9.])/(?:Users|home)/[A-Za-z0-9_.-]+/',
    'private data root': rb'/data/(?:private|nas)/',
    'private network URL': rb'https?://(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)(?::\d+)?(?:/|\b)',
    'private key': rb'-----BEGIN [A-Z ]*PRIVATE KEY-----',
    'credential assignment': rb'(?i)(?:api_key|access_token|secret_key)\s*[:=]\s*[\x22\x27][A-Za-z0-9_/-]{20,}',
}
ALLOWED_SUFFIXES = {'.md', '.py', '.json', '.js', '.html', '.css', '.toml', '.txt', '.yml', ''}
IGNORED = {'.git', '__pycache__', '.pytest_cache', '.venv', 'build', 'dist'}
def generated(path):
    return any(x in IGNORED or x.endswith('.egg-info') for x in path.parts)


def verify(root, refresh=False):
    """Reject any unreviewed file before hashing or producing an archive."""
    manifest_path = root/'PUBLIC_FILES.json'
    manifest_bytes = manifest_path.read_bytes()
    for label, pattern in PATTERNS.items():
        if re.search(pattern, manifest_bytes):
            raise ValueError(label+' detected in the manifest')
    manifest = json.loads(manifest_bytes)
    names = [row['path'] for row in manifest['files']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate allowlist entries.')
    expected = set(names) | {'PUBLIC_FILES.json'}
    for path in root.rglob('*'):
        if path.is_symlink() and not generated(path.relative_to(root)):
            raise ValueError('Symlinks are outside the public boundary.')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and not generated(p.relative_to(root))}
    if actual != expected:
        raise ValueError(f'Allowlist mismatch: extra={sorted(actual-expected)}, missing={sorted(expected-actual)}')
    for row in manifest['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe allowlist path.')
        path = root/relative
        if path.is_symlink() or path.suffix not in ALLOWED_SUFFIXES:
            raise ValueError('Unexpected artifact type: '+row['path'])
        data = path.read_bytes()
        data.decode('utf-8')
        for label, pattern in PATTERNS.items():
            if re.search(pattern, data):
                raise ValueError(f'{label} detected in {row["path"]}')
        digest = hashlib.sha256(data).hexdigest()
        if refresh:
            row.update(sha256=digest, bytes=len(data))
        elif digest != row['sha256'] or len(data) != row['bytes']:
            raise ValueError('Content differs from reviewed manifest: '+row['path'])
    if refresh:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(f'PASS: {len(names)} allowlisted files; hashes and common leak checks passed.')


def main():
    """Check this package, optionally refreshing reviewed file hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    verify(Path(__file__).resolve().parents[1], args.refresh)


if __name__ == '__main__':
    main()
