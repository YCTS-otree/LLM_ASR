"""Explicit, opt-in official weight download. Never called by application startup."""
import argparse
import hashlib
import json
from pathlib import Path
from qwen_spec import MODEL_SPECS, WEIGHT_FILE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Explicitly permit official model downloads')
    parser.add_argument('--model', choices=tuple(MODEL_SPECS), default='2b')
    parser.add_argument('--path', type=Path)
    parser.add_argument('--source', choices=['modelscope', 'huggingface'], default='modelscope')
    args = parser.parse_args()
    if not args.download:
        parser.error('Use --download to explicitly allow the model download')
    spec = MODEL_SPECS[args.model]
    args.path = args.path or spec['path']
    print(f'Downloading official {spec["model_id"]} via {args.source} to {args.path}', flush=True)
    patterns = ['*.safetensors', '*.json', '*.jinja', 'merges.txt', 'LICENSE', 'README.md']
    if args.source == 'modelscope':
        from modelscope import snapshot_download
        snapshot_download(spec['model_id'], revision='master', local_dir=str(args.path),
                          allow_patterns=patterns, max_workers=4)
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(spec['model_id'], revision=spec['revision'], local_dir=args.path, token=False,
                          allow_patterns=patterns, max_workers=2)
    digest = hashlib.sha256()
    with (args.path / WEIGHT_FILE).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != spec['sha256']:
        raise RuntimeError('Official model SHA256 mismatch')
    (args.path / 'local_manifest.json').write_text(json.dumps({
        'model_id': spec['model_id'], 'verified_hf_revision': spec['revision'], 'source': args.source,
        'weight_sha256': digest.hexdigest()
    }, indent=2), encoding='utf-8')
    print('Download complete; SHA256 verified. Application can load this path offline.', flush=True)


if __name__ == '__main__':
    main()
