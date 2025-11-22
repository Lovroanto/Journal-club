import os
import re
import ollama
from langchain_ollama import OllamaLLM


# =========================================================
# USER-SELECTABLE QUESTION FORMATS
# =========================================================
QUESTION_FORMATS = {
    "A": r"(?P<id>[0-9]+)\s*\.\s*(?=[A-ZÉÈÊÀÇÎÔÛÂ])",  # 1. Calculer
    "B": r"(?P<id>[0-9]+)\s*\)\s*(?=[A-ZÉÈÊÀÇÎÔÛÂ])",  # 1) Calculer
    "C": r"(?P<id>[a-z])\s*\)\s*(?=[A-ZÉÈÊÀÇÎÔÛÂ])",   # a) Calculer
    "D": r"(?P<id>[0-9]+)\.\s*(?=[A-ZÉÈÊÀÇÎÔÛÂ])",     # 1.Calculer
    "E": r"(?P<id>[a-z])\.\s*(?=[A-ZÉÈÊÀÇÎÔÛÂ])",      # a. Calculer
}


def choose_question_format():
    print("\n=== SELECT QUESTION FORMAT ===")
    print("Choose the format used in your document:\n")
    print("A → 1. Calculer")
    print("B → 1) Calculer")
    print("C → a) Calculer")
    print("D → 1.Calculer")
    print("E → a. Calculer")
    print()

    while True:
        choice = input("Enter format letter (A/B/C/D/E): ").strip().upper()
        if choice in QUESTION_FORMATS:
            return QUESTION_FORMATS[choice]
        print("Invalid choice. Try again.")


# =========================================================
# 1) BUILD QUESTION DETECTOR REGEX
# =========================================================
def build_question_regex(format_pattern):
    return re.compile(
        rf"""
        (?P<tag>
            (?<!\d)             # avoid matching decimals
            (?<![A-Za-z])       # avoid inside words
            {format_pattern}
        )
        """,
        re.VERBOSE,
    )


# =========================================================
# 2) QUESTION CLASS
# =========================================================
class Question:
    def __init__(self, abs_order, placement, snippet):
        self.abs_order = abs_order
        self.placement = placement
        self.snippet = snippet  # first 20 chars

    def __repr__(self):
        return f"Question(abs_order={self.abs_order}, placement={self.placement}, snippet='{self.snippet}')"


def convert_id_to_int(q_id):
    """Converts numeric or letter ID to integer placement."""
    if q_id.isdigit():
        return int(q_id)
    else:
        return ord(q_id.lower()) - ord("a") + 1


# =========================================================
# 3) DETECT QUESTION STARTS
# =========================================================
def detect_question_starts(text, question_regex):
    ids = []
    questions = []
    out = []
    last_idx = 0
    abs_order = 0

    for match in question_regex.finditer(text):
        start = match.start()
        q_id = match.group("id")
        abs_order += 1
        placement = convert_id_to_int(q_id)
        snippet = text[start:start + 20]

        questions.append(Question(abs_order, placement, snippet))
        ids.append(q_id)

        out.append(text[last_idx:start])
        out.append(f"<QUESTION_START>{match.group('tag')}")
        last_idx = match.end("tag")

    out.append(text[last_idx:])
    return "".join(out), questions


# =========================================================
# 4) ASK AI LAST 15 CHARS (OPTIONAL)
# =========================================================
def ask_ai_last_chars(excerpt, model="open-mistral-nemo"):
    """
    Calls an AI model (Ollama, etc.) to extract the last 15 characters
    of the last question contained in the excerpt.
    """

    prompt = (
        "Below is a text excerpt that contains the end of a question from an exercise, "
        "possibly followed by the beginning of the next exercise.\n\n"
        "Your task:\n"
        "👉 Extract ONLY the last 15 characters of the LAST QUESTION in the excerpt.\n"
        "❌ Do NOT include any text belonging to the next exercise.\n"
        "❌ Do NOT include any question number like '1.' or '2)'.\n\n"
        "Return ONLY those 15 characters, nothing else.\n\n"
        "EXCERPT:\n"
        + excerpt
    )

    result = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    answer = result["message"]["content"].strip()

    # safety trim in case LLM returns extra spaces/newlines
    signature = answer[-15:]

    # PRINTING FOR USER TO VERIFY
    print("\n==============================")
    print(" AI-EXTRACTED LAST 15 CHARS ")
    print("==============================")
    print(f"→ '{signature}'")
    print("==============================\n")

    return signature


# =========================================================
# 5) DETECT EXERCISE ENDS BY PLACEMENT ORDER
# =========================================================
def detect_exercise_ends(text, questions, use_ai=False, model="open-mistral-nemo"):
    """
    Inserts <EXERCISE_END> before first question of a new exercise.
    Logic: If a question has smaller placement than previous, it starts a new exercise.
    If use_ai=True, uses ask_ai_last_chars for more precise boundary.
    """
    if not questions:
        return text

    exercise_boundaries = []

    for i in range(1, len(questions)):
        if questions[i].placement < questions[i - 1].placement:
            exercise_boundaries.append((questions[i - 1], questions[i]))  # (last prev, first next)

    modified = text
    for last_q, next_q in reversed(exercise_boundaries):

        print(f"*** Detected exercise boundary: last placement '{last_q.placement}' → next placement '{next_q.placement}'")

        # Extract region between last question and next question
        start_prev = modified.find(last_q.snippet) + len(last_q.snippet)
        start_next = modified.find(next_q.snippet)
        excerpt = modified[start_prev:start_next]

        if use_ai:
            signature = ask_ai_last_chars(excerpt, model=model)
            sig_index = modified.rfind(signature)
            if sig_index != -1:
                insert_pos = sig_index + len(signature)
                modified = modified[:insert_pos] + "\n<EXERCISE_END>\n" + modified[insert_pos:]
                continue

        # fallback: insert before next question
        modified = modified[:start_next] + "<EXERCISE_END>\n" + modified[start_next:]

    return modified


# =========================================================
# 6) PROCESS ONE CHUNK
# =========================================================
def process_chunk(text, question_regex, use_ai=False, model="open-mistral-nemo"):
    q_tagged, questions = detect_question_starts(text, question_regex)
    final = detect_exercise_ends(q_tagged, questions, use_ai=use_ai, model=model)
    return final


# =========================================================
# 7) RUN TAGGER
# =========================================================
def run_tagger(input_dir, output_dir, model=None, use_ai=False):
    format_pattern = choose_question_format()
    question_regex = build_question_regex(format_pattern)

    os.makedirs(output_dir, exist_ok=True)
    chunk_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".txt")
    ])

    for f in chunk_files:
        print("Processing:", f)
        original = open(f).read()
        tagged = process_chunk(original, question_regex, use_ai=use_ai, model=model)
        out_f = os.path.join(output_dir, os.path.basename(f))
        open(out_f, "w").write(tagged)

    print("Tagging completed (placement-based with optional AI refinement).")



# =========================================================
# 5) REMAINING UTILITIES (UNCHANGED)
# =========================================================

def collect_all_text(tagged_dir):
    files = sorted([
        os.path.join(tagged_dir, f)
        for f in os.listdir(tagged_dir)
        if f.endswith(".txt")
    ])
    return "\n".join(open(f).read() for f in files)


def split_exercises(full_text, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    exercises = full_text.split("<EXERCISE_END>")
    for idx, ex in enumerate(exercises, start=1):
        ex_folder = os.path.join(out_dir, f"ex{idx:02d}")
        os.makedirs(ex_folder, exist_ok=True)

        # Split intro
        parts = ex.split("<QUESTION_START>")
        intro = parts[0].strip()
        questions = parts[1:]

        open(os.path.join(ex_folder, "intro.txt"), "w").write(intro)

        for q_i, q_text in enumerate(questions, start=1):
            q_path = os.path.join(ex_folder, f"q{q_i:02d}.txt")
            open(q_path, "w").write(q_text.strip())

    print("Exercise splitting complete.")
