from pathlib import Path


def load(name: str, directory: str = "./templates") -> str:
    matches = list(Path(directory).glob(name))
    if not matches:
        matches = list(Path(directory).glob(f"*{name}*"))
    if not matches:
        raise FileNotFoundError(f"No prompt matched {name!r} in {directory}")
    if len(matches) > 1:
        names = ", ".join(str(p) for p in matches)
        print(f"(Warning) Prompt name is ambiguous: {names}")

    return matches[0].read_text(encoding="utf-8")


def default() -> str:
    return "{query}\n\n{documents}"
