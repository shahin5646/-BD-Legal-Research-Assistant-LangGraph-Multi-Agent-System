"""Source cross-verification for legal research.

After an interview is completed, this module extracts key legal claims
and checks them against the accumulated search context (including supporting
law and counter-argument search results). Claims identified with contradicting
evidence get flagged with [⚠️ Cross-Check: Disputed] markers in the section text.
"""

import logging

from pydantic import BaseModel, Field
from langchain.messages import SystemMessage, HumanMessage

from src.utils.models import get_llm

logger = logging.getLogger(__name__)


class ClaimExtraction(BaseModel):
    """Key legal claims extracted from an interview for verification."""

    claims: list[str] = Field(
        description="List of 2-4 key factual/legal claims that should be checked. "
                    "Each claim should be a single, verifiable statement."
    )


class VerificationResult(BaseModel):
    """Result of cross-checking a single claim against search results."""

    claim: str = Field(description="The original claim being checked.")
    status: str = Field(
        description="'verified' if supporting evidence confirmed, "
                    "'disputed' if contradicting evidence found, "
                    "'unverified' if no clear evidence either way."
    )
    evidence: str = Field(
        default="",
        description="Brief summary of supporting or contradicting evidence found."
    )


class VerificationReport(BaseModel):
    """Full verification report for claims from an interview."""

    results: list[VerificationResult] = Field(
        description="Verification results for each extracted claim."
    )
    summary: str = Field(
        description="One-sentence overall assessment of source consistency."
    )


_EXTRACT_PROMPT = SystemMessage(content="""You are a legal fact-checker.

From the following legal interview transcript, extract 2-3 key factual/legal claims
that are most important to verify. Focus on:

1. Specific legal provisions cited (Acts, Sections, Ordinances)
2. Claims about what the law requires, permits, or prohibits
3. Statements about legal precedents or court rulings

Each claim should be a single, concise statement.""")


_VERIFY_PROMPT = SystemMessage(content="""You are a legal verification expert for Bangladesh law.

Given the accumulated search results (which include both supporting sources and counter-argument searches),
evaluate the following legal claim:

Determine if the claim is:
- "verified" — the search results confirm and support the claim
- "disputed" — the search results contradict, override, or challenge the claim
- "unverified" — the search results do not clearly support or contradict

Provide brief evidence for your determination.

Claim to verify: {claim}""")


def extract_claims(interview_text: str) -> list[str]:
    """Extract key claims from an interview transcript."""
    try:
        model = get_llm()
        structured = model.with_structured_output(ClaimExtraction)
        result = structured.invoke([
            _EXTRACT_PROMPT,
            HumanMessage(content=f"Extract claims from:\n\n{interview_text}"),
        ])
        if result and result.claims:
            logger.info("Extracted %d claims for cross-check", len(result.claims))
            return result.claims
    except Exception:
        logger.warning("Claim extraction failed")
    return []


def verify_claim(claim: str, search_context: str) -> VerificationResult:
    """Check a single claim against search results."""
    try:
        model = get_llm()
        structured = model.with_structured_output(VerificationResult)
        prompt = _VERIFY_PROMPT.content.format(claim=claim)
        result = structured.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Search results:\n\n{search_context}"),
        ])
        if result:
            logger.info("Claim cross-checked: '%s' → %s", claim[:50], result.status)
            return result
    except Exception:
        logger.warning("Cross-check failed for claim: %s", claim[:50])

    return VerificationResult(
        claim=claim,
        status="unverified",
        evidence="Cross-check could not be completed.",
    )


def format_verification_markers(section_text: str, results: list[VerificationResult]) -> str:
    """Add [⚠️ Disputed] or [✅ Verified] markers to a section based on verification results."""
    annotated = section_text

    for result in results:
        if result.status == "disputed":
            annotated += f"\n\n> ⚠️ **Disputed**: {result.claim}\n> *Evidence*: {result.evidence}"
        elif result.status == "verified":
            annotated += f"\n\n> ✅ **Verified**: {result.claim}"

    return annotated

