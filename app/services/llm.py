from openai import OpenAI
from ..config import settings

client = OpenAI()

def summarize_text(text: str) -> str:
    resp = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a concise educational summarizer."},
            {"role": "user", "content": f"Summarize this for quick learning:\n\n{text}"}
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def make_revision_notes(text: str) -> str:
    prompt = (
        "Create 5-8 concise bullet-point revision notes or flashcards from this summary. "
        "Keep each point short and exam-oriented.\n\n" + text
    )
    resp = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": "You generate crisp study notes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()