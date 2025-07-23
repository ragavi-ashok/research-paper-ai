# research-paper-ai

This repository runs a set of experiments probing how various LLMs answer INTRA‑style financial decision questions.  
Because LLM outputs are non‑deterministic, your results may differ from the included `.xlsx` workbooks.

---

## Contents

- **intra-prompt.txt**  
  Base INTRA survey prompt (100 trials).

- **chatgpt_non-reasoning.py**  
  Queries ChatGpt non-reasoning API 

- **chatgpt_reasoning.py**  
 .Queries ChatGpt reasoning API 

- **gemini.py**  
  Queries Gemini 2.0 Flash API.

- **deepseek.py**  
  Queries DeepSeek R1 API.

- **\*.xlsx**  
  Raw responses saved by each script:
  - `ChatGPT_4o_answers.xlsx`  
  - `ChatGPT_4.1_answers.xlsx`  
  - `ChatGPT_4.5-preview_answers.xlsx`  
  - `ChatGPT_o3-mini_answers.xlsx`  
  - `Deepseek_R1_answers_1to50.xlsx`  
  - `Deepseek_R1_answers_51to100.xlsx`  
  - `Gemini_2.0flash_answers.xlsx`  

---

## Per‑script Documentation

Each script in this repo has its own README file named `<script_name>_README.md` in the repository root.  
Those per‑script READMEs include:
- Detailed description  
- Prerequisites and dependencies  

Please refer to the individual `<script_name>_README.md` files for script‑specific details.

---

## Prerequisites

- **Python 3.8+**  
- Clone this repo and install dependencies:  
  ```bash
  git clone https://github.com/your-username/research-paper-ai.git
  cd research-paper-ai
  pip install -r requirements.txt
Place your API keys in environment variables before running any script:
export OPENAI_API_KEY="…"
export GEMINI_API_KEY="…"
export DEEPSEEK_API_KEY="…"
Usage

Run any script with:

python <script_name>.py

For example:
python chatgpt_non-reasoning.py
Results will be written to the corresponding Excel workbook (e.g. ChatGPT_4o_answers.xlsx).

Note: Because LLM outputs evolve and are sampled randomly, your results likely won’t exactly match the files committed here.
