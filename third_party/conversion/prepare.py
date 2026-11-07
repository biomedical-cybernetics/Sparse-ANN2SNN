"""Install conversion dependencies."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent

def prepare(name, destination, archive_path=None):
    spec = json.loads((HERE / 'components.json').read_text())[name]
    if archive_path:
        data = Path(archive_path).read_bytes()
    else:
        with urllib.request.urlopen(spec['archive_url'], timeout=60) as response:
            data = response.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        members = {m.name.split('/', 1)[1]: m for m in archive
                   if m.isfile() and '/' in m.name}
        outputs = []
        for record in spec['files']:
            relative = Path(record['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid component path')
            original = archive.extractfile(members[record['path']]).read()
            lines = original.decode('utf-8').splitlines(keepends=True)
            for edit in reversed(record['edits']):
                lines[edit['start']:edit['end']] = edit['replacement']
            result = ''.join(lines).encode('utf-8')
            target = destination / name / relative
            if target.exists() and target.read_bytes() != result:
                raise FileExistsError('Refusing to replace different source: ' + str(target))
            outputs.append((target, result))
        for target, result in outputs:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(result)
    print(name + ': prepared ' + str(len(outputs)) + ' source files')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=['dcgs', 'parallel', 'all'], default='all')
    parser.add_argument('--destination', type=Path, default=HERE.parent / 'sources')
    parser.add_argument('--archive', type=Path, help='Local pinned archive; one component only')
    args = parser.parse_args()
    if args.archive and args.component == 'all':
        parser.error('--archive requires a single component')
    for name in ['dcgs', 'parallel'] if args.component == 'all' else [args.component]:
        prepare(name, args.destination, args.archive)

if __name__ == '__main__':
    main()
