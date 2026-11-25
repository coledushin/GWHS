#!/usr/bin/env python3
"""Render exam reports with Quarto (report.qmd), one per */*/report.yml.

    python3 render.py                       # every exam
    python3 render.py "US/June Regents"     # just the exams whose folder matches
    python3 render.py --suffix .quarto      # write us_..._report.quarto.html (side-by-side checks)
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))


def render(cfg_path, suffix=''):
    with open(cfg_path, encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)
    stem, ext = os.path.splitext(cfg['output'])
    target = os.path.join(ROOT, stem + suffix + ext)
    tmp = 'report.html'   # Quarto writes next to report.qmd; moved into place below
    print(f'--- {cfg_path} ---', flush=True)
    # The page <title> comes from a metadata file (quarto's -M flag doesn't set it).
    with tempfile.NamedTemporaryFile('w', suffix='.yml', delete=False, encoding='utf-8') as meta:
        yaml.safe_dump({'title': cfg['title']}, meta, allow_unicode=True)
    try:
        result = subprocess.run(
            ['quarto', 'render', 'report.qmd', '--output', tmp, '--metadata-file', meta.name],
            cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, 'REPORT_CONFIG': cfg_path})
    finally:
        os.unlink(meta.name)
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        print(f'  FAILED: {cfg_path}', file=sys.stderr)
        return False
    shutil.move(os.path.join(ROOT, tmp), target)
    print(f'  Done → {os.path.relpath(target, ROOT)}  ({os.path.getsize(target) // 1024} KB)')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('match', nargs='?', default='', help='only exams whose folder contains this')
    ap.add_argument('--suffix', default='', help='added before .html in the output name')
    args = ap.parse_args()
    os.chdir(ROOT)
    configs = sorted(p for p in glob.glob('*/*/report.yml') if args.match in p)
    if not configs:
        sys.exit(f'No report.yml found matching "{args.match}"')
    ok = [render(p, args.suffix) for p in configs]
    print(f'\n{sum(ok)} of {len(ok)} rendered')
    sys.exit(0 if all(ok) else 1)


if __name__ == '__main__':
    main()
