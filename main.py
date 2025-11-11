import os
import fitz  # PyMuPDF
from langchain_ollama import OllamaLLM

# === CONFIGURATION ===
BASE_DIR = os.path.expanduser("~/ai_data/Journal_Club/First")
MAIN_ARTICLE_PATH = os.path.join(BASE_DIR, "main", "s41567-025-02854-4.pdf")
SUMMARY_PATH = os.path.join(BASE_DIR, "summary.txt")
PLAN_PATH = os.path.join(BASE_DIR, "plan.txt")

# === LOAD AND EXTRACT PDF TEXT ===
def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    doc.close()
    print(f"✅ Extracted {len(text)} characters from PDF.")
    return text

# === INITIALIZE OLLAMA MODEL ===
llm = OllamaLLM(model="llama3.1")

# === ANALYSIS PROMPTS ===
def analyze_article(text):
    print("🧠 Analyzing article content with Ollama...")

    summary_prompt = f"""
    You are a scientific assistant. Summarize the following scientific article clearly and concisely.
    Focus on the background, main hypothesis, methodology, and key results.

    ARTICLE TEXT:
    {text[:12000]}  # truncate long papers for first pass
    """
    summary = llm.invoke(summary_prompt)

    plan_prompt = f"""
    Based on this article summary, create a structured plan for a journal club presentation.
    - Identify 2–3 key concepts or notions that must be explained before presenting the results.
    - Suggest which figures or results should be shown (if any are mentioned).
    - Propose how to divide the talk into slides (intro, background, methods, results, discussion).
    - List 3–5 references or notions that should be further researched.

    ARTICLE SUMMARY:
    {summary}
    """
    plan = llm.invoke(plan_prompt)

    return summary, plan

# === MAIN EXECUTION ===
def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    text = extract_text_from_pdf(MAIN_ARTICLE_PATH)
    summary, plan = analyze_article(text)

    # Save results
    with open(SUMMARY_PATH, "w") as f:
        f.write(summary)
    with open(PLAN_PATH, "w") as f:
        f.write(plan)

    print(f"\n✅ Summary saved to: {SUMMARY_PATH}")
    print(f"✅ Plan saved to: {PLAN_PATH}")

if __name__ == "__main__":
    main()