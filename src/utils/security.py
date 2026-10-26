"""Security Layer — input sanitization, PII detection/masking, output validation.

Provides a SecurityPipeline that is wired into graph nodes to:
  1. Block prompt injection attempts in user input and retrieved web sources
  2. Detect and mask PII (including Bangladesh mobile numbers and NIDs)
  3. Validate LLM responses for leaked secrets and harmful exploits
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input Sanitization
# ---------------------------------------------------------------------------

class InputSanitizer:
    """Block prompt injection and clean risky input."""

    # Common jailbreak / injection phrases
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(all\s+)?previous",
        r"new\s+instructions\s*:",
        r"system\s*prompt",
        r"---\s*end\s*(of)?\s*prompt",
        r"pretend\s+you\s+are",
        r"act\s+as\s+(if\s+)?you",
        r"bypass\s+(all\s+)?restrictions",
        r"reveal\s+(your|the)\s+(system|instructions|prompt)",
        r"you\s+are\s+now\s+(DAN|jailbroken)",
    ]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def check(self, text: str) -> tuple[bool, Optional[str]]:
        """Return (is_safe, reason)."""
        if not text:
            return True, None
        for pattern in self.patterns:
            if pattern.search(text):
                logger.warning("Prompt injection detected: %s", pattern.pattern)
                return False, "Blocked: potential prompt injection detected"
        return True, None

    def clean(self, text: str) -> str:
        """Sanitize risky syntax while preserving valid markdown elements."""
        if not text:
            return ""
        # Neutralize template injection syntax
        cleaned = text.replace("{{", "{ {").replace("}}", "} }")
        # Strip excessive leading/trailing whitespace
        return cleaned.strip()


# ---------------------------------------------------------------------------
# PII Detection & Masking
# ---------------------------------------------------------------------------

class PIIDetector:
    """Find and mask personal data, supporting Bangladesh and international formats."""

    PATTERNS = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        # Bangladeshi mobile: +880 1712 345678, +880-1712-345678, 01712-345678, 01712345678
        "bd_phone": re.compile(r"(?:\+?880[-\s]?|0)1[3-9]\d{2}[-\s]?\d{6}\b"),
        # US/Intl standard phone with explicit delimiters (avoids matching plain numbers)
        "phone": re.compile(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b"),
        # Context-aware NID to prevent false positives on amounts or case numbers
        "nid": re.compile(
            r"(?i)\b(?:nid|national\s+id(?:entity)?(?:\s+no\.?)?)\s*[:#-]?\s*(\d{10}|\d{13}|\d{17})\b"
        ),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    }

    MASK_MAP = {
        "email": "[EMAIL REDACTED]",
        "bd_phone": "[PHONE REDACTED]",
        "phone": "[PHONE REDACTED]",
        "nid": "[NID REDACTED]",
        "ssn": "[SSN REDACTED]",
        "credit_card": "[CARD REDACTED]",
    }

    def detect(self, text: str) -> dict[str, list[str]]:
        """Return PII types found, with matches."""
        found = {}
        for pii_type, pattern in self.PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                # If pattern has capturing groups (e.g. NID digits), extract strings
                extracted = [m if isinstance(m, str) else m[0] for m in matches]
                found[pii_type] = extracted
        return found

    def mask(self, text: str) -> str:
        """Replace all PII with redaction tags without breaking surrounding content."""
        masked = text
        for pii_type, pattern in self.PATTERNS.items():
            if pii_type == "nid":
                # Only mask the digits part of the NID match
                def _mask_nid(m):
                    full = m.group(0)
                    digits = m.group(1)
                    return full.replace(digits, self.MASK_MAP["nid"])
                masked = pattern.sub(_mask_nid, masked)
            else:
                masked = pattern.sub(self.MASK_MAP[pii_type], masked)
        return masked


# ---------------------------------------------------------------------------
# Output Validation
# ---------------------------------------------------------------------------

class OutputValidator:
    """Check LLM output for PII leaks and harmful exploits."""

    HARMFUL_PATTERNS = [
        re.compile(r"here('s| is) (how|the way) to (hack|steal|exploit|breach)", re.I),
        re.compile(
            r"(?i)\bpassword\s*(?:is\s*[:=]|\s*[:=]|\s+is\s+)\s*[\"']?([A-Za-z0-9!@#$%^&*_-]{4,})[\"']?"
            r"(?!\s+(?:as|a|an|the|in|by|under|for|to|with|of|from|that|which))\b"
        ),
        re.compile(r"(?i)\bapi[_\s]?key\s*[:=]\s*[\"']?[A-Za-z0-9_-]{10,}"),
    ]

    def __init__(self):
        self.pii_detector = PIIDetector()

    def validate(self, output: str) -> tuple[str, list[str]]:
        """Return (cleaned_output, warnings).

        Redacts specific harmful snippets or masks PII rather than destroying
        the entire research report.
        """
        warnings = []

        # 1. Mask PII leaks
        pii_found = self.pii_detector.detect(output)
        if pii_found:
            output = self.pii_detector.mask(output)
            warnings.append(f"PII masked in output: {list(pii_found.keys())}")
            logger.warning("PII detected in LLM output: %s", list(pii_found.keys()))

        # 2. Check and redact harmful exploit content
        for pattern in self.HARMFUL_PATTERNS:
            if pattern.search(output):
                output = pattern.sub("[Blocked: potentially harmful content]", output)
                warnings.append("Harmful content detected and redacted")
                logger.warning("Harmful content redacted in LLM output")

        return output, warnings


# ---------------------------------------------------------------------------
# Security Pipeline (main entry point)
# ---------------------------------------------------------------------------

class SecurityPipeline:
    """Runs input checks and output validation."""

    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.pii_detector = PIIDetector()
        self.output_validator = OutputValidator()

    def check_input(self, text: str) -> tuple[bool, str, list[str]]:
        """Return (is_allowed, cleaned_text, notes)."""
        notes = []

        # 1. Block prompt injection
        is_safe, reason = self.sanitizer.check(text)
        if not is_safe:
            return False, "", [reason]

        # 2. Clean delimiters
        cleaned = self.sanitizer.clean(text)

        # 3. Mask PII before hitting the LLM
        pii_found = self.pii_detector.detect(cleaned)
        if pii_found:
            cleaned = self.pii_detector.mask(cleaned)
            notes.append(f"Input PII masked: {list(pii_found.keys())}")

        return True, cleaned, notes

    def check_output(self, text: str) -> tuple[str, list[str]]:
        """Return (cleaned_output, warnings)."""
        return self.output_validator.validate(text)



# Shared instance for all nodes
security = SecurityPipeline()