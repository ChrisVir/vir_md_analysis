from pathlib import Path


def ls(path, name_only=True, suffix='', prefix='', infix=''):
    p = Path(path)
    if not p.exists():
        return []
    if name_only:
        return [f.name for f in p.glob(f'*{suffix}') if f.is_file()
                and f.name.startswith(prefix) and infix in f.name]

    return [f for f in p.glob(f'*{suffix}') if f.is_file()
            and f.name.startswith(prefix) and infix in f.name]
