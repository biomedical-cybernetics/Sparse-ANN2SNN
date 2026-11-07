#!/usr/bin/env python3
"""Install the pinned RIA modules used by DeiT pruning."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent


def normalize(value):
    return value.decode('utf-8').replace('\r\n', '\n').rstrip('\n') + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, help='Use an existing pinned lib/ directory instead of downloading')
    p.add_argument('--output-dir', type=Path, default=HERE.parent/'sources'/'ria')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    manifest = json.loads((HERE/'ria.json').read_text())
    for record in manifest['files']:
        target = a.output_dir/'lib'/record['name']
        if a.check_only:
            if not target.is_file():
                raise ValueError(f'Missing dependency: {target}')
            compile(target.read_bytes(), str(target), 'exec')
            print('Found', record['name'])
            continue
        if a.source_dir:
            raw = (a.source_dir/record['name']).read_bytes()
        else:
            with urlopen(record['url'], timeout=30) as response:
                raw = response.read()
        text = normalize(raw)
        lines = text.splitlines(True)
        for change in reversed(record['patch']):
            lines[change['start']:change['end']] = change['replacement']
        content = ''.join(lines).encode()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f'Refusing to replace a modified dependency: {target}')
        target.write_bytes(content)
        print('Installed', record['name'])


if __name__ == '__main__':
    main()
