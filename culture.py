"""The culture layer: turns a Taglish query into something the model understands.

"lolo singing videoke noche buena" becomes:
  - visual text:  "an elderly grandfather singing people singing karaoke ..." (a few variants, averaged)
  - people boost: photos containing whoever you've linked to "lolo"
  - date window:  Dec 24 to Dec 25, any year
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG = Path(__file__).parent / "config"
STOPWORDS = {"mga", "ng", "ang", "si", "ni", "kay", "sa", "nina", "sina"}
MAX_VARIANTS = 3


def norm(text: str) -> str:
    """Lowercase and treat hyphens as spaces, so halo-halo == halo halo."""
    return " ".join(str(text).lower().replace("-", " ").split())


def _load(name):
    path = CONFIG / name
    if not path.exists():
        return {}
    out = {}
    for k, v in (yaml.safe_load(path.read_text()) or {}).items():
        out.setdefault(norm(k), v)
    return out


@dataclass
class Parsed:
    original: str
    variants: list[str]
    persons: set[str] = field(default_factory=set)
    date_windows: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)  # Filipino terms recognized in the query

    def explain(self) -> str:
        lines = [f"  searching for: {' | '.join(self.variants)}"]
        lines += [f"  {n}" for n in self.notes]
        return "\n".join(lines)


class Culture:
    def __init__(self, person_names: list[str]):
        self.lexicon = _load("lexicon.yaml")
        self.holidays = _load("holidays.yaml")
        raw_kin = _load("kinship.yaml")

        # Kinship: explicit names from kinship.yaml, plus auto-link anyone named
        # with the term itself in Photos ("Lolo", "Tita Baby", ...).
        self.kin = {term: set(names or []) for term, names in raw_kin.items()}
        for name in person_names:
            words = set(norm(name).split())
            for term in self.kin:
                if term in words:
                    self.kin[term].add(name)

        # Match longest terms first so "noche buena" wins over a single word.
        all_terms = set(self.lexicon) | set(self.holidays) | set(self.kin)
        self.terms = sorted(all_terms, key=lambda t: (-len(t.split()), -len(t)))

    def parse(self, query: str, keep_original: bool = False) -> Parsed:
        text = norm(query)
        slots: list[str] = []
        for term in self.terms:
            pattern = rf"(?<![\w-]){re.escape(term)}(?![\w-])"
            if re.search(pattern, text):
                text = re.sub(pattern, f" \x00{len(slots)}\x00 ", text, count=1)
                slots.append(term)

        p = Parsed(original=query, variants=[], terms=list(slots))
        for term in slots:
            if term in self.kin and self.kin[term]:
                p.persons |= self.kin[term]
                p.notes.append(f"{term} -> people: {', '.join(sorted(self.kin[term]))}")
            if term in self.holidays:
                start, end = self.holidays[term]
                p.date_windows.append((start, end))
                p.notes.append(f"{term} -> dates {start} to {end}")

        n = max([len(self.lexicon.get(t, [])) for t in slots] + [1])
        for j in range(min(n, MAX_VARIANTS)):
            out = text
            for i, term in enumerate(slots):
                opts = self.lexicon.get(term) or [term]
                out = out.replace(f"\x00{i}\x00", opts[j % len(opts)])
            words = [w for w in out.split() if w not in STOPWORDS]
            p.variants.append(" ".join(words) or query)
        if keep_original and slots:
            # Also keep the user's own words: CLIP may already know them
            # (lumpia) or read them as text/logos in the photo (Fruitas).
            p.variants.append(" ".join(w for w in norm(query).split() if w not in STOPWORDS))
        p.variants = list(dict.fromkeys(p.variants))  # dedupe, keep order
        return p
