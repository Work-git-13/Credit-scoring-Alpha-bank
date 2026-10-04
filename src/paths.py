from pathlib import Path

def find_project_root(start=None):
    """Ищем корень проекта из notebooks или любой вложенной папки."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / 'requirements.txt').is_file() and (candidate / 'src').is_dir():
            return candidate
    raise FileNotFoundError('Открой Jupyter из папки credit-scoring или её notebooks.')
