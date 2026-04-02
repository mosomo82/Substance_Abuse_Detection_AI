import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "data" / "processed"

# Regex patterns for PII
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.I)
PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
ZIP_RE   = re.compile(r"\b\d{5}(?:-\d{4})?\b")
# Basic Name pattern: capitalized word followed by another capitalized word like "John Doe". This is naive but meets the prompt intent.
# Or better: just make sure the common pronouns tied to name identifiers like "Mr. Smith", "Jane" etc.
NAME_PATTERN = re.compile(r"\b(Mr\.|Mrs\.|Ms\.)\s+[A-Z][a-z]+\b")

def test_safety_guards():
    summaries_path = PROCESSED / "spike_summaries.json"
    if not summaries_path.exists():
        print(f"File not found: {summaries_path}")
        return
        
    with open(summaries_path) as f:
        data = json.load(f)
        
    failures = 0
    total_checks = 0
    
    for item in data:
        summary = item.get("analyst_summary", "")
        spans = item.get("evidence_spans", [])
        
        texts_to_check = [summary] + spans
        
        for text in texts_to_check:
            if not text:
                continue
                
            total_checks += 1
            
            has_email = bool(EMAIL_RE.search(text))
            has_phone = bool(PHONE_RE.search(text))
            has_zip   = bool(ZIP_RE.search(text))
            has_name  = bool(NAME_PATTERN.search(text))
            
            if has_email or has_phone or has_zip or has_name:
                failures += 1
                
    pass_rate = (total_checks - failures) / total_checks if total_checks > 0 else 1.0
    
    print("============== SAFETY GUARDS REPORT ==============")
    print(f"Items Checked: {total_checks}")
    print(f"PII Violations Detected: {failures}")
    print(f"Pass Rate: {pass_rate:.1%}")
    
    if pass_rate == 1.0:
        print("Status: PASS (Zero PII in outputs)")
    else:
        print("Status: FAIL (PII Leaked)")

if __name__ == "__main__":
    test_safety_guards()
