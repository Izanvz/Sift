"""Self-critique scoring used in Sift's answer quality loop."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CritiqueScore:
    faithfulness: float      # Is the answer grounded in retrieved chunks? (0-10)
    completeness: float      # Does it answer the full question? (0-10)
    citation_quality: float  # Are citations accurate and present? (0-10)

    @property
    def overall(self) -> float:
        return (self.faithfulness + self.completeness + self.citation_quality) / 3

    def needs_rewrite(self) -> bool:
        """Faithfulness below 6.0 always triggers a rewrite."""
        return self.faithfulness < 6.0


def parse_critique_response(raw: str) -> CritiqueScore:
    """Parse LLM critique JSON output into a structured score.

    Expected format:
        {"faithfulness": 8.0, "completeness": 7.5, "citation_quality": 9.0}
    """
    import json
    import re

    match = re.search(r"\{[^}]+\}", raw)
    if not match:
        return CritiqueScore(faithfulness=0.0, completeness=0.0, citation_quality=0.0)

    data = json.loads(match.group())
    return CritiqueScore(
        faithfulness=float(data.get("faithfulness", 0)),
        completeness=float(data.get("completeness", 0)),
        citation_quality=float(data.get("citation_quality", 0)),
    )
