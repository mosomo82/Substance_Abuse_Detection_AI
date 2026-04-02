import sys
from pathlib import Path
import pandas as pd
import random
import re

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from processing.preprocess_posts import SLANG_LEXICON, preprocess_post
from classifiers.rule_based_classifier import classify_post

DATA = ROOT / "data"
IN_CSV = DATA / "raw" / "posts_classified.csv"

# Invert SLANG_LEXICON: Canonical -> List of Slang
CANONICAL_TO_SLANG = {}
for slang, canon in SLANG_LEXICON.items():
    if canon not in CANONICAL_TO_SLANG:
        CANONICAL_TO_SLANG[canon] = []
    CANONICAL_TO_SLANG[canon].append(slang)

def find_high_risk_posts(df, n=50):
    # Only want original text that contains our canonical terms and was marked high risk 
    # either by logic or we can just pick posts that have canonical drug terms and distress.
    # To be safe, we'll preprocess, classify, and pick 50 true 'high'.
    high_candidates = []
    
    # Take a sample to speed up finding 50
    sample_df = df.sample(min(2000, len(df)), random_state=42)
    
    for idx, row in sample_df.iterrows():
        text = row["review"] if "review" in row else row["text"]
        rec = preprocess_post(text, post_id=row.get("post_id", idx))
        
        # Must contain at least one canonical term we can replace
        has_replaceable = any(c in rec["processed_text"] for c in CANONICAL_TO_SLANG)
        if not has_replaceable:
            continue
            
        clf_result = classify_post(rec)
        if clf_result["risk_level"] == "high":
            high_candidates.append(text)
            if len(high_candidates) >= n:
                break
                
    return high_candidates

def mutate_to_slang(text):
    # Replace any canonical word with a random slang associated with it
    mutated = text
    for canon, slangs in CANONICAL_TO_SLANG.items():
        # word boundary replacement (case insensitive)
        pattern = re.compile(r'\b' + re.escape(canon) + r'\b', re.IGNORECASE)
        # We replace each match independently to add variety
        def repl(m):
            return random.choice(slangs)
        mutated = pattern.sub(repl, mutated)
    return mutated

def run_tests():
    print("Loading posts for metamorphic testing...")
    if not IN_CSV.exists():
        print(f"File not found: {IN_CSV}")
        return
        
    df = pd.read_csv(IN_CSV).dropna(subset=["review" if "review" in pd.read_csv(IN_CSV).columns else "text"])
    
    print("Finding 50 high-risk baseline posts with canonical terms...")
    base_texts = find_high_risk_posts(df, n=50)
    
    if not base_texts:
        print("Could not find any suitable high risk posts!")
        return
        
    print(f"Found {len(base_texts)} posts. Running substitutions...")
    
    passes = 0
    failures = 0
    
    for i, text in enumerate(base_texts):
        mutated = mutate_to_slang(text)
        
        rec = preprocess_post(mutated, post_id=i)
        clf_result = classify_post(rec)
        
        if clf_result["risk_level"] == "high":
            passes += 1
        else:
            failures += 1
            
    total = passes + failures
    pass_rate = passes / total if total > 0 else 0
    
    print("============== METAMORPHIC TEST REPORT ==============")
    print(f"Total Evaluated: {total}")
    print(f"Passed: {passes}")
    print(f"Failed: {failures}")
    print(f"Pass Rate: {pass_rate:.1%}")
    
    if pass_rate >= 0.80:
        print("Status: PASS")
    else:
        print("Status: FAIL (Below 80% threshold)")

if __name__ == "__main__":
    run_tests()
