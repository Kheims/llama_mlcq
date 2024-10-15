import json
import logging
import tqdm
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                    handlers=[
                        logging.FileHandler('model_results.log'),
                        logging.StreamHandler()
                    ])

MAX_CHARS = 20000
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct" 

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

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
    prompt = f"""
    You are a code analysis assistant. Please analyze the following code snippet and identify any code smell between:
    "feature_envy", "long_method", "blob", "data_class". 
    Additionally, rate the severity of the code smell as: "none", "minor," "moderate," or "severe."
    Code snippet:
    ```
    {code_snippet}
    ```
    Provide your response in the exact format: "Smell: <name>, Severity: <severity>"
    Do not add any other thing to the response.
    """
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    generated_text = response[len(prompt):].strip()
    return generated_text

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
                correct_answer = f"{entry['smell']}, {entry['severity']}"
                logging.info(f"Processed snippet {entry['unique_id']}: Model Output: {result}, Correct: {correct_answer}")
            
            if (i + 1) % batch_size == 0:
                save_results(results, results_filepath)
            
            pbar.update(1)
    
    save_results(results, results_filepath)
    print("Smell detection completed. Results saved to 'results.json'.")

if __name__ == '__main__':
    json_file_path = 'updated_data.json'
    process_json(json_file_path)