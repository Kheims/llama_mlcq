from openai import OpenAI
from openai import OpenAIError, RateLimitError
import openai
import json
import logging
import tqdm
import os 
import time

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                    handlers=[
                        logging.FileHandler('model_results.log'),
                        logging.StreamHandler()
                    ])

MAX_CHARS = 20000

def truncate_snippet(snippet):
    """Truncate the code snippet to stay within the character limit."""
    return snippet[:MAX_CHARS] if len(snippet) > MAX_CHARS else snippet

def load_existing_results(filepath):
    """Load existing results from a JSON file if it exists."""
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            return json.load(file)
    return []

def save_results(results, filepath):
    """Save the results to a JSON file."""
    with open(filepath, "w") as file:
        json.dump(results, file, indent=4)

def detect_smell_and_severity(code_snippet):
    code_snippet = truncate_snippet(code_snippet)
    # Create the prompt with the code snippet
    prompt = f"""
    You are a code analysis assistant. Please analyze the following code snippet and identify any code smell between :
    "feature_envy", "long_method", "blob", "data_class". 
    Additionally, rate the severity of the code smell as: "none", "minor," "moderate," or "severe."

    Code snippet:
    ```
    {code_snippet}
    ```

    Provide your response in the exact format: "Smell: <name>, Severity: <severity>"
    Do not add any other thing to the response.
    """
    retries = 5
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in software engineering and code analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            message = response.choices[0].message.content
            return message
        except RateLimitError as e:
            wait_time = (2 ** attempt) * 5
            logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds.")
            time.sleep(wait_time)
    raise Exception("Failed to complete requests after multiple retries.")

def process_json(file_path, results_filepath='results.json', batch_size=20):
    existing_results = load_existing_results(results_filepath)
    processed_ids = {result['unique_id'] for result in existing_results}

    results = existing_results  
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    with tqdm.tqdm(total=len(data), desc="Processing snippets") as pbar:
        for i, entry in enumerate(data):
            if entry['unique_id'] in processed_ids:
                pbar.update(1)
                continue

            code_snippet = entry['code_snippet']
            result = detect_smell_and_severity(code_snippet)
            if result is not None:
                results.append({
                    'unique_id': entry['unique_id'],
                    'smell_and_severity': result
                })

                correct_answer = entry.get('smell_and_severity', 'N/A')
                logging.info(f"Processed snippet {entry['unique_id']}: Model Output: {result}, Correct: {entry['smell']}, {entry['severity']}")

            if (i + 1) % batch_size == 0:
                save_results(results, results_filepath)
            
            pbar.update(1)    
            
        print("Smell detection completed. Results saved to 'smell_detection_results.json'.")

if __name__ == '__main__':
    json_file_path = 'updated_data.json'
    process_json(json_file_path)
