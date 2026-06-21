from pathlib import Path


def find_project_root(
    start_path: Path | None = None, markers: tuple = ("pyproject.toml", ".git")
) -> Path:
    path = (start_path or Path.cwd()).resolve()
    for parent in [path, *path.parents]:
        if any((parent / marker).exists() for marker in markers):
            return parent
    raise FileNotFoundError("No se encontró la raíz del proyecto")


ROOT_PATH      = find_project_root(Path(__file__))
DATA_PATH      = ROOT_PATH / "data"
RAW_PATH       = DATA_PATH / "raw"
PROCESSED_PATH = DATA_PATH / "processed"
EXTERNAL_PATH  = DATA_PATH / "external_clean"


def project_path(*parts: str) -> Path:
    """Build paths relative to the project root."""
    return ROOT_PATH.joinpath(*parts)
