from openai import OpenAI
from ..config import settings
import os 
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_text(text: str) -> str:
    resp = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a concise educational summarizer. Write summaries that are easy to read aloud and understand. Use simple, clear sentences. Avoid complex vocabulary, abbreviations, or technical jargon unless absolutely necessary. Write in a conversational tone that flows well when spoken."},
            {"role": "user", "content": f"Create a clear, simple summary of this text that would be easy to read aloud and translate. Use simple language and short sentences:\n\n{text}"}
        ],
    )
    return resp.choices[0].message.content.strip()

def make_revision_notes(text: str) -> str:
    prompt = (
        "Create 5-8 simple revision notes from this summary. "
        "Use simple, everyday language that is easy to read aloud and translate. "
        "Avoid complex vocabulary, abbreviations, or technical terms. "
        "Write each point as a clear, short sentence that flows naturally when spoken. "
        "Format with simple bullet points using • symbols. "
        "Make the content accessible for students of all levels.\n\n" + text
    )
    resp = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": "You generate simple, clear revision notes that are easy to read aloud and translate. Use everyday language, avoid jargon, and write in a way that flows naturally when spoken. Focus on clarity and simplicity."},
            {"role": "user", "content": prompt}
        ],
    )
    return resp.choices[0].message.content.strip()