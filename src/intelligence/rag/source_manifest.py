"""Source manifest loader — Sprint LLM-2B.2.

Loads ``data/rag/sources_manifest.yaml`` into typed records the RAG
ingestor can consume. Exposes a search/filter surface so the
retrieval layer can prefer high-authority sources, exclude biased
authors, or filter by language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SourceRecord:
    id: str
    type: str               # paper | report | data | educational
    title: str
    authors: tuple[str, ...]
    year: int
    venue: str = ""
    url: str = ""
    language: str = "en"
    authority: int = 3      # 1..5
    license: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "url": self.url,
            "language": self.language,
            "authority": self.authority,
            "license": self.license,
            "tags": list(self.tags),
        }


def load_manifest(path: str | Path) -> list[SourceRecord]:
    """Parse a YAML manifest into SourceRecord objects.

    Defers the YAML import so callers without PyYAML installed get a
    clean ImportError pointing to the dep instead of a module-level
    crash.
    """
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to read sources_manifest.yaml — "
            "pip install pyyaml"
        ) from exc
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out: list[SourceRecord] = []
    for entry in data.get("sources", []):
        out.append(
            SourceRecord(
                id=str(entry["id"]),
                type=str(entry.get("type", "educational")),
                title=str(entry.get("title", "")),
                authors=tuple(entry.get("authors", []) or []),
                year=int(entry.get("year", 0) or 0),
                venue=str(entry.get("venue", "") or ""),
                url=str(entry.get("url", "") or ""),
                language=str(entry.get("language", "en")),
                authority=int(entry.get("authority", 3)),
                license=str(entry.get("license", "") or ""),
                tags=tuple(entry.get("tags", []) or []),
            )
        )
    return out


def filter_sources(
    sources: list[SourceRecord],
    *,
    type: Optional[str] = None,
    language: Optional[str] = None,
    min_authority: int = 1,
    exclude_tags: Optional[list[str]] = None,
) -> list[SourceRecord]:
    """Filter a list — used by the retriever to whitelist high-trust sources
    on sensitive narrative paths (e.g. exclude_tags=['biased-author'])."""
    out = []
    excl = set(exclude_tags or [])
    for s in sources:
        if type and s.type != type:
            continue
        if language and s.language != language:
            continue
        if s.authority < min_authority:
            continue
        if excl & set(s.tags):
            continue
        out.append(s)
    return out


__all__ = ["SourceRecord", "filter_sources", "load_manifest"]
