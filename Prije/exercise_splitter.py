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
            (?<!\d)
            (?<![A-Za-z])
            {format_pattern}
        )
        """,
        re.VERBOSE,
    )

# =========================================================
# 2) QUESTION CLASS WITH CHUNK INFO
# =========================================================
class Question:
    def __init__(self, abs_order, placement, snippet, chunk_id):
        self.abs_order = abs_order        # absolute order of question in document
        self.placement = placement        # placement according to user-selected numbering
        self.snippet = snippet            # first 20 characters for identification
        self.chunk_id = chunk_id          # which chunk it was found in

    def __repr__(self):
        return f"Question(order={self.abs_order}, place={self.placement}, chunk={self.chunk_id}, snippet='{self.snippet}')"

def convert_id_to_int(q_id):
    """Convert numeric or letter ID to integer placement."""
    if q_id.isdigit():
        return int(q_id)
    else:
        return ord(q_id.lower()) - ord("a") + 1

# =========================================================
# 3) DETECT QUESTION STARTS
# =========================================================
def detect_question_starts(text, question_regex, chunk_id):
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

        questions.append(Question(abs_order, placement, snippet, chunk_id))
        ids.append(q_id)

        out.append(text[last_idx:start])
        out.append(f"<QUESTION_START>{match.group('tag')}")
        last_idx = match.end("tag")

    out.append(text[last_idx:])
    return "".join(out), questions

# =========================================================
# 4) ASK AI LAST 15 CHARS
# =========================================================
def ask_ai_last_chars(excerpt, model="open-mistral-nemo"):
    """
    Calls Ollama to extract the last 15 characters of the last question in excerpt.
    """
    prompt = (
        "Below is a text excerpt that contains the end of a question from an exercise, "
        "possibly followed by the beginning of the next exercise.\n\n"
        "Your task:\n"
        "👉 Extract ONLY the last 15 characters of the LAST QUESTION in the excerpt.\n"
        "❌ Do NOT include any text belonging to the next exercise.\n"
        "❌ Do NOT include any question number like '1.' or '2)'.\n\n"
        "Return ONLY those 15 characters, nothing else.\n\n"
        "EXCERPT:\n" + excerpt
    )

    result = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    answer = result["message"]["content"].strip()
    signature = answer[-15:]

    print("\n==============================")
    print(" AI-EXTRACTED LAST 15 CHARS ")
    print("==============================")
    print(f"→ '{signature}'")
    print("==============================\n")

    return signature

# =========================================================
# 5) DETECT EXERCISE ENDS USING AI
# =========================================================
def detect_exercise_ends(text, questions, model="open-mistral-nemo"):
    """
    Inserts <EXERCISE_END> before the first question of a new exercise.
    Logic:
    - consecutive questions with decreasing placement → new exercise
    - uses AI to find exact end of last question
    """
    if not questions:
        return text

    modified = text
    for i in range(1, len(questions)):
        prev_q = questions[i - 1]
        curr_q = questions[i]

        if curr_q.placement < prev_q.placement or curr_q.chunk_id != prev_q.chunk_id:
            print(f"*** Detected exercise boundary: last question '{prev_q.snippet[:10]}...' → next question '{curr_q.snippet[:10]}...'")
            
            # Extract excerpt between chunks if necessary
            start_idx = modified.find(prev_q.snippet) + len(prev_q.snippet)
            end_idx = modified.find(curr_q.snippet)
            excerpt = modified[start_idx:end_idx]

            # AI helps find last 15 chars
            signature = ask_ai_last_chars(excerpt, model=model)
            sig_index = modified.rfind(signature)
            if sig_index == -1:
                print("⚠️ Signature not found — fallback before next question")
                sig_index = end_idx

            insert_pos = sig_index + len(signature)
            modified = modified[:insert_pos] + "\n<EXERCISE_END>\n" + modified[insert_pos:]

    return modified

# =========================================================
# 6) PROCESS ONE CHUNK
# =========================================================
def process_chunk(text, question_regex, chunk_id, model="open-mistral-nemo"):
    q_tagged, questions = detect_question_starts(text, question_regex, chunk_id)
    final = detect_exercise_ends(q_tagged, questions, model=model)
    return final

# =========================================================
# 7) RUN TAGGER
# =========================================================
def run_tagger(input_dir, output_dir, model="open-mistral-nemo"):
    format_pattern = choose_question_format()
    question_regex = build_question_regex(format_pattern)

    os.makedirs(output_dir, exist_ok=True)
    chunk_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".txt")
    ])

    for chunk_idx, f in enumerate(chunk_files, start=1):
        print("Processing:", f)
        original = open(f).read()
        tagged = process_chunk(original, question_regex, chunk_idx, model=model)
        out_f = os.path.join(output_dir, os.path.basename(f))
        open(out_f, "w").write(tagged)

    print("Tagging completed (AI-assisted, chunk-aware).")



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
