"""Unit tests for source cross-verification."""

import pytest
from src.utils.verification import (
    ClaimExtraction,
    VerificationResult,
    VerificationReport,
    format_verification_markers,
)


class TestVerificationModels:
    def test_claim_extraction(self):
        ce = ClaimExtraction(claims=["Section 2 of Companies Act requires registration"])
        assert len(ce.claims) == 1

    def test_empty_claims(self):
        ce = ClaimExtraction(claims=[])
        assert ce.claims == []

    def test_verification_result_verified(self):
        vr = VerificationResult(
            claim="Companies Act 1994 Section 2 requires registration",
            status="verified",
            evidence="Confirmed by bdlaws.minlaw.gov.bd",
        )
        assert vr.status == "verified"

    def test_verification_result_disputed(self):
        vr = VerificationResult(
            claim="No registration needed for sole proprietorship",
            status="disputed",
            evidence="Bangladesh Trade Organisation Ordinance 1961 requires trade licence",
        )
        assert vr.status == "disputed"

    def test_verification_result_unverified(self):
        vr = VerificationResult(
            claim="Some unclear legal position",
            status="unverified",
            evidence="",
        )
        assert vr.status == "unverified"

    def test_verification_report(self):
        results = [
            VerificationResult(claim="Claim 1", status="verified", evidence="Ok"),
            VerificationResult(claim="Claim 2", status="disputed", evidence="Wrong"),
        ]
        report = VerificationReport(
            results=results,
            summary="Mixed reliability.",
        )
        assert len(report.results) == 2


class TestFormatVerificationMarkers:
    def test_disputed_marker_added(self):
        section = "## Some Section\n\nSome content."
        results = [
            VerificationResult(
                claim="Registration is optional",
                status="disputed",
                evidence="Act says otherwise",
            )
        ]
        annotated = format_verification_markers(section, results)
        assert "⚠️ **Disputed**" in annotated
        assert "Registration is optional" in annotated
        assert "Act says otherwise" in annotated

    def test_verified_marker_added(self):
        section = "## Section\n\nContent."
        results = [
            VerificationResult(
                claim="Section 2 applies",
                status="verified",
                evidence="Confirmed",
            )
        ]
        annotated = format_verification_markers(section, results)
        assert "✅ **Verified**" in annotated

    def test_unverified_not_added(self):
        """Unverified claims should not add markers."""
        section = "## Section\n\nContent."
        results = [
            VerificationResult(
                claim="Some claim",
                status="unverified",
                evidence="",
            )
        ]
        annotated = format_verification_markers(section, results)
        assert annotated == section  # no markers added

    def test_multiple_markers(self):
        section = "## Section\n\nContent."
        results = [
            VerificationResult(claim="Claim A", status="disputed", evidence="E1"),
            VerificationResult(claim="Claim B", status="verified", evidence="E2"),
            VerificationResult(claim="Claim C", status="unverified", evidence=""),
        ]
        annotated = format_verification_markers(section, results)
        assert "⚠️ **Disputed**" in annotated
        assert "✅ **Verified**" in annotated
        assert annotated.count("⚠️") == 1  # only 1 disputed
        assert annotated.count("✅") == 1  # only 1 verified

    def test_empty_results_unchanged(self):
        section = "## Section\n\nContent."
        annotated = format_verification_markers(section, [])
        assert annotated == section
