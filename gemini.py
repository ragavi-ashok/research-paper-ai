

"""
LLM Financial Decision Simulation — Gemini 2.0 Flash
-----------------------------------------------------

This script sends a 14-question financial decision-making prompt to Google's Gemini 2.0 Flash model across 100 trials with each trial in a new temp chat.
All 14 questions are asked in a single prompt per trial. 
The model's structured responses (e.g., 'A', 'B', or numeric values) are parsed and stored in a tidy CSV file.

Configuration:
- Replace the placeholder API key with your actual Gemini API key.
- Model used: `"models/gemini-2.0-flash"`
- Adjust `n_trials` to control how many times the full prompt is submitted.
- Each trial is sent as a new, stateless conversation.
- Ensure the prompt text is available in the same folder as the script as a file named `intra-prompt.txt`.
    This file should contain the full 14-question text in plain English. The script reads this file using:
    questions = load_prompt("intra-prompt.txt")

Output:
- A CSV file (e.g., `gemini_answers.csv`) containing columns: `Trial, Q1, ..., Q14`
- One row per trial
- Logs are saved in the `logs/` folder with timestamped filenames

Response Format Warning:
The script expects output in the form:
    1. A  
    2. 110  
    ...
If Gemini changes its formatting (e.g., switches to `"Q1 = A"` or paragraph-style responses),
you must update the `parse_multi_answer_reply()` function accordingly.

API Key:
Do NOT hardcode your real Gemini key into public scripts.
Use environment variables or secure secrets management in production environments.


Installation Requirements:
- Python 3.8 or later
- Install dependencies using pip:

    pip install google-generativeai pandas

- Optional (if managing API keys securely):

    pip install python-dotenv
"""


import os
import re
import time
import pandas as pd
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai

# Setup Gemini API
genai.configure(api_key="ENTER YOUR API KEY")

# genai.configure(api_key="ENTER YOUR API KEY")
model = genai.GenerativeModel("models/gemini-2.0-flash")

# Configurations
n_trials = 1
outfile = "gemini_answers.csv"
dry_run_outfile = "dry_run.csv"
max_workers = 5
dry_run = False

# Create a folder for logs
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_folder, f"gemini_flash_run_{n_trials}_{timestamp}.log")

# Set up logging to file and console
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

def load_prompt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()

# Extracts a clean answer (either a single letter or a number)
def extract_answer(text):
    if not text:
        return None
    text = text.strip()
    if re.match(r"^[AaBb]$", text):
        return text.upper()
    match = re.search(r"\$?([\d,]+(?:\.\d+)?)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None

# Submits prompt to Gemini or returns a dummy response in dry_run mode
def ask_gemini(prompt_text):
    if dry_run:
        logging.info(f"[DRY RUN] Prompt preview: {prompt_text[:50]}...")
        return "A" if "letter" in prompt_text else "100"
    try:
        response = model.generate_content(
            prompt_text,
            generation_config=genai.types.GenerationConfig(temperature=0.7),
        )
        return response.text.strip()
    except Exception as e:
        logging.warning(f"Gemini error: {e}")
        time.sleep(5)
        return None

# Parses Gemini response in "1. Answer" format
# The model may not always give the answer in this format. Its good to be cautious here.
def parse_multi_answer_reply(reply_text, trial):
    answers = {}
    lines = reply_text.strip().splitlines()
    numbered_pattern = re.compile(r"^\s*(\d+)\.\s*(.+)$")

    for line in lines:
        match = numbered_pattern.match(line)
        if match:
            q_num = int(match.group(1))
            ans_text = match.group(2).strip()
            parsed = extract_answer(ans_text)
            logging.info(f"[TRIAL {trial}][PARSER] Q{q_num}: '{ans_text}' ➝ {parsed}")
            answers[q_num] = parsed
        else:
            logging.debug(f"[TRIAL {trial}][PARSER] Skipping unmatched line: {line}")
    return answers

# Runs a single trial of all 14 questions
def process_single_trial(trial):
    row = {"Trial": trial}
    logging.info(f"[TRIAL {trial}] Submitting questions...")
    # Load the prompt that contain the survey questions
    questions = load_prompt("intra-prompt.txt")
    reply = ask_gemini(questions)
    logging.info(f"[TRIAL {trial}] Response:\n{reply}")
    answers = parse_multi_answer_reply(reply, trial)
    for i in range(1, 15):
        row[f"Q{i}"] = answers.get(i, None)
    return row

# Executes multiple trials in parallel using threads
def run_trials(n_trials):
    all_results = []
    trials = list(range(1, n_trials + 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_trial, t): t for t in trials}
        for future in as_completed(futures):
            try:
                row = future.result()
                all_results.append(row)
            except Exception as e:
                logging.error(f"Trial failed: {e}")
    return all_results

# Run experiment and gather responses
results = run_trials(n_trials)

# Convert results to DataFrame
df = pd.DataFrame(results)
df = df.sort_values(by="Trial").reset_index(drop=True)
ordered_columns = ["Trial"] + [f"Q{i}" for i in range(1, 15)]
df = df[ordered_columns]

# Save results to CSV
output_file = dry_run_outfile if dry_run else outfile
df.to_csv(output_file, index=False, float_format="%.2f")

# Final logs
logging.info(f"✅ Results saved to: {output_file}")
logging.info(f"📄 Log file: {log_file}")