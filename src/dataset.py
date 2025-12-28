import json
from typing import Iterator, Dict, Optional

def load_jsonl(path: str, lexem_filter: Optional[str] = None) -> Iterator[Dict]:
    """
    Stream a JSONL file line by line.

    Parameters
    ----------
    path : str
        Path to JSONL file.
    lexem_filter : str, optional
        Only yield samples with this lexem.

    Yields
    ------
    dict
        Sample dictionary with keys like 'id', 'sentence', 'span', etc.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            if lexem_filter is not None and sample.get("lexem") != lexem_filter:
                continue
            yield sample
