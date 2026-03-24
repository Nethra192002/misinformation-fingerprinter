import ollama
import re

MODEL_NAME = "llama3.1"

CLAIM_EXTRACTION_PROMPT = """You are a claim extraction system. Your job is to extract the single core factual claim from any input text.

Rules:
- Output exactly one sentence
- Remove all opinions, emotions, platform metadata, usernames, hashtags, URLs
- Keep only verifiable factual assertions
- If the input is already a clean claim, return it as-is
- Never add commentary, explanation, or preamble
- Never start with "The claim is" or similar phrases
- Output only the claim itself

Text: {text}

Claim:"""


def extract_claim(text: str) -> str:
    text = _preprocess(text)

    if not text or len(text.split()) < 3:
        return ""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": CLAIM_EXTRACTION_PROMPT.format(text=text)}],
            options={"temperature": 0.0}
        )
        claim = response["message"]["content"].strip()
        claim = _postprocess(claim)
        return claim

    except Exception as e:
        print(f"Ollama error: {e}")
        return _fallback_clean(text)


def _preprocess(text: str) -> str:
    text = text.strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _postprocess(claim: str) -> str:
    lines = claim.strip().split("\n")
    claim = lines[0].strip()
    claim = re.sub(r"^(claim:|the claim is|core claim:)", "", claim, flags=re.IGNORECASE).strip()
    if claim and not claim.endswith("."):
        claim += "."
    return claim


def _fallback_clean(text: str) -> str:
    text = _preprocess(text)
    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences[0] + "." if sentences else text[:200]


if __name__ == "__main__":
    test_inputs = [
        "BREAKING: At least one killed, several injured during shootout with #CharlieHebdo attack suspects - @BBCBreaking http://t.co/abc123",
        "Police say 12 people were killed after armed gunmen stormed the headquarters of French weekly publication Charlie Hebdo in Paris.",
        "RT @SkyNews: Witnesses say Ferguson shooting victim had hands up when shot by police officer http://bit.ly/xyz",
        "I can't believe what's happening in Ottawa right now, gunman opened fire at Parliament killing a soldier",
        "https://www.bbc.com/news/world-europe-30710883"
    ]

    print("Testing Input Normalizer\n")
    for i, text in enumerate(test_inputs):
        print(f"Input {i+1}: {text[:80]}...")
        claim = extract_claim(text)
        print(f"Claim:   {claim}")
        print()