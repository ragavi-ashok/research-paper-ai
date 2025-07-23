"""
LLM Financial Decision Simulation — DeepSeek R1
------------------------------------------------

This script sends a 14-question financial decision-making prompt to the DeepSeek R1 model across 100 trials with each trial in a new temp chat.
All 14 questions are asked together in a single prompt per trial. 
The model's answers (e.g., 'A', 'B', or numeric values) are parsed and saved into a structured CSV format.

Configuration:
- Replace the placeholder API key with your actual DeepSeek API key.
- Set the appropriate model endpoint for DeepSeek R1.
- Adjust `n_trials` to control the number of full-prompt trials.
- Each trial is stateless and independent.
- Ensure the prompt text is available in the same folder as the script as a file named `intra-prompt.txt`.
    This file should contain the full 14-question text in plain English. The script reads this file using:
    questions = load_prompt("intra-prompt.txt")

Output:
- A CSV file (e.g., `deepseek_r1_answers.csv`) with columns: `Trial, Q1, ..., Q14`
- One row per trial
- Log files saved in the `logs/` folder, timestamped for traceability

Response Format Warning:
The script expects clean responses like:
    1. A  
    2. 110  
If the model returns answers in a different structure (e.g., paragraphs or alternative numbering),
update the `parse_multi_answer_reply()` function to ensure accurate parsing.

API Key:
Do **not** include your real API key directly in shared or public scripts.
Use environment variables or secure key management instead.


Installation Requirements:
- Python 3.8 or later
- Install required libraries with pip:

    pip install requests pandas

- Optional (for API key management):

    pip install python-dotenv
"""


import os
import re
import time
import logging
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai 
from openai import OpenAI

# Setup Deepseek API
client = OpenAI(
    api_key="ENTER YOUR API KEY",  # 🔁 Replace this with your real key
    base_url="https://api.deepseek.com/v1"
)
model_name = "deepseek-reasoner"

# Configurations
n_trials = 1
outfile = "deepseek_answers.csv"
dry_run_outfile = "dry_run.csv"
max_workers = 5
dry_run = False

# Create a folder for logs
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_folder, f"deepseek_run_{n_trials}_{timestamp}.log")

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

# --- DeepSeek API Call ---
def ask_deepseek(prompt_text):
    if dry_run:
        logging.info(f"[DRY RUN] Prompt preview: {prompt_text[:50]}...")
        return "A" if "letter" in prompt_text else "100"
    else:
        for _ in range(3):  # Retry logic
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logging.warning(f"DeepSeek error: {e}")
                time.sleep(5)

# Parses Deepseek response in "1. Answer" format
# The model may not always give the answer in this format. Its good to be cautious here.
def parse_multi_answer_reply(reply_text, trial):
    answers = {}
    lines = [line.strip() for line in reply_text.strip().splitlines() if line.strip()]
    logging.info(f"[TRIAL {trial}][PARSER] Detected {len(lines)} response lines")

    for i, line in enumerate(lines[:14], start=1):  # Only take the first 14 answers
        parsed = extract_answer(line)
        logging.info(f"[TRIAL {trial}][PARSER] Q{i}: '{line}' ➝ {parsed}")
        answers[i] = parsed

    return answers

# Runs a single trial of all 14 questions
def process_single_trial(trial):
    row = {"Trial": trial}
    logging.info(f"[TRIAL {trial}] Submitting questions...")
    # Load the prompt that contain the survey questions
    questions = load_prompt("intra-prompt.txt")
    reply = ask_deepseek(questions)
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