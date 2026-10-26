import re

from pydantic import BaseModel

UNSAFE_PATTERNS = {
    "prompt_injection": r"ignore (all |any )?(previous|prior|above) instructions|admin mode|system prompt",
    "bulk_action": r"(refund|cancel) (every|all) (order|customer)s?",
    "data_exfiltration": r"customer list|all (customer )?emails|export (all )?(customers|users)|password",
}


class SafetyVerdict(BaseModel):
    safe: bool
    flags: list[str]


def assess_request(text: str) -> SafetyVerdict:
    flags = [name for name, pattern in UNSAFE_PATTERNS.items() if re.search(pattern, text, re.IGNORECASE)]
    return SafetyVerdict(safe=not flags, flags=flags)
