from __future__ import annotations

import re

from app.schemas.transform import Finding


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


class PIICheckService:
    def evaluate(self, raw_prompt: str) -> list[Finding]:
        findings: list[Finding] = []

        email_matches = EMAIL_RE.findall(raw_prompt)
        phone_matches = PHONE_RE.findall(raw_prompt)
        ssn_matches = SSN_RE.findall(raw_prompt)

        if len(email_matches) >= 2:
            findings.append(
                Finding(
                    type="pii",
                    severity="high",
                    code="email_list_detected",
                    message="This prompt appears to include multiple email addresses or contact records.",
                )
            )
        elif len(email_matches) == 1:
            findings.append(
                Finding(
                    type="pii",
                    severity="medium",
                    code="email_detected",
                    message="This prompt appears to include an email address.",
                )
            )

        if phone_matches:
            findings.append(
                Finding(
                    type="pii",
                    severity="medium",
                    code="phone_detected",
                    message="This prompt appears to include a phone number.",
                )
            )

        if ssn_matches:
            findings.append(
                Finding(
                    type="pii",
                    severity="high",
                    code="sensitive_identifier_detected",
                    message="This prompt appears to include a highly sensitive identifier.",
                )
            )

        return findings
