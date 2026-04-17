from __future__ import annotations

from app.schemas.transform import Finding


class ComplianceCheckService:
    def evaluate(self, raw_prompt: str) -> list[Finding]:
        text = raw_prompt.lower()
        findings: list[Finding] = []

        if any(term in text for term in ("medical advice", "legal advice", "financial advice")):
            findings.append(
                Finding(
                    type="compliance",
                    severity="medium",
                    code="regulated_advice",
                    message="This prompt appears to request regulated professional advice.",
                )
            )

        if any(term in text for term in ("bypass authentication", "exploit vulnerability", "steal credentials", "malware")):
            findings.append(
                Finding(
                    type="compliance",
                    severity="high",
                    code="security_sensitive_request",
                    message="This prompt appears to request security-sensitive or abusive instructions.",
                )
            )

        if (
            any(term in text for term in ("confidential", "internal only", "private"))
            and any(term in text for term in ("customer data", "employee data", "prospect list", "client list"))
        ):
            findings.append(
                Finding(
                    type="compliance",
                    severity="high",
                    code="confidential_business_data",
                    message="This prompt appears to involve confidential business data.",
                )
            )

        return findings
