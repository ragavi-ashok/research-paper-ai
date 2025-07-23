"""
LLM Financial Decision Simulation — ChatGPT 
---------------------------------------------------

This script sends a 14-question financial decision-making prompt to OpenAI's (this below has `o3-mini` model)
across 100 trials with each trial in a new temp chat. All 14 questions are asked together in a single prompt per trial.
It captures the model’s structured responses (e.g., 'A', 'B', or numeric values),parses them, and saves the results in a CSV file.

Configuration:
- Replace the placeholder API key with your actual OpenAI API key.
- Set `model_name = "o3-mini"
- The same script can be used with "o1" by simply updating the model name.
- Adjust `n_trials` to control how many runs of the prompt will be executed.
- Each trial is sent as a new, stateless conversation.
- Ensure the prompt text is available in the same folder as the script as a file named `intra-prompt.txt`.
    This file should contain the full 14-question text in plain English. The script reads this file using:
    questions = load_prompt("intra-prompt.txt")

Output:
- A CSV file (e.g., `o3-mini.csv`) containing columns `Trial, Q1, ..., Q14`
- One row per trial
- Logs saved in the `logs/` folder with timestamped filenames

Response Format Warning:
The script expects model output in the form:
    1. A
    2. 110
If OpenAI modifies this output format (e.g., plain numbered lines or narrative replies),
you must update the `parse_multi_answer_reply()` function accordingly.

API Key:
Do NOT hardcode your real OpenAI key into public scripts.
Use environment variables or secrets management for production usage.


Installation Requirements:
- Python 3.8 or later
- Install dependencies using pip:

    pip install openai pandas

- Optional (if managing API key via `.env` file):

    pip install python-dotenv
"""

import os
import re
import time
import logging
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, OpenAIError, RateLimitError, APIError
from openai import OpenAI

# Setup OpenAI API
client = OpenAI(api_key="ENTER YOUR API KEY")
max_workers = 5
dry_run = False
n_trials = 100
model_name = "o3-mini"
outfile = "o3-mini.csv"

dry_run_outfile = "dry_run.csv"

# Create a folder for logs
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_folder, f"chatgpt_o3_mini_run_{n_trials}_{timestamp}.log")

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

# ChatGPT o3 Mini
def ask_gpt(prompt_text):
    if dry_run:
        logging.info(f"[DRY RUN] Prompt preview: {prompt_text[:50]}...")
        return "A" if "letter" in prompt_text else "100"
    else:
        for _ in range(3):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt_text} 
                    ]
                )
                return response.choices[0].message.content.strip()
            except RateLimitError:
                logging.warning("⚠️ Rate limited. Waiting 20s...")
                time.sleep(20)
            except APIError as e:
                logging.warning(f"⚠️ API error: {e}")
                time.sleep(5)
            except OpenAIError as e:
                logging.error(f"❌ OpenAI error: {e}")
                raise


# Parses Chatgpt o3 response in "1. Answer" format
# The model may not always give the answer in this format. Its good to be cautious here.
def parse_multi_answer_reply(reply_text, trial):
    """
    Strictly parses only lines like '1: B' or '2: 110'.
    Ignores all other content (e.g., explanations).
    Only keeps the first valid answer per question 1–14.
    """
    answers = {}
    lines = [line.strip() for line in reply_text.strip().splitlines() if line.strip()]
    numbered_pattern = re.compile(r"^(\d{1,2})\s*[:]\s*(.+)$")

    for line in lines:
        match = numbered_pattern.match(line)
        if match:
            q_num = int(match.group(1))
            if 1 <= q_num <= 14 and q_num not in answers:
                ans_text = match.group(2).strip()
                parsed = extract_answer(ans_text)
                answers[q_num] = parsed
                logging.info(f"[TRIAL {trial}][PARSER] Q{q_num}: '{ans_text}' ➝ {parsed}")
        if len(answers) == 15:
            break  

    return answers

# Runs a single trial of all 14 questions
def process_single_trial(trial):
    row = {"Trial": trial}
    logging.info(f"[TRIAL {trial}] Submitting questions...")
    # Load the prompt that contain the survey questions
    questions = load_prompt("intra-prompt.txt")
    reply = ask_gpt(questions)
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