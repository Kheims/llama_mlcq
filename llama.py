import transformers
import torch
import gc
import json
import logging
import os
import tqdm
import bitsandbytes as bnb  

torch.cuda.empty_cache()
gc.collect()

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "load_in_4bit": True,  
        "bnb_4bit_quant_type": "nf4",  
        "device_map": "auto" 
    },
)

MAX_CHARS = 40000

examples = [
{
        "code_snippet": "  @Override\n  public boolean incrementToken() throws IOException {\n    for(;;) {\n\n      if (!remainingTokens.isEmpty()) {\n        // clearAttributes();  // not currently necessary\n        restoreState(remainingTokens.removeFirst());\n        return true;\n      }\n\n      if (!input.incrementToken()) return false;\n\n      int len = termAtt.length();\n      if (len==0) return true; // pass through zero length terms\n      \n      int firstAlternativeIncrement = inject ? 0 : posAtt.getPositionIncrement();\n\n      String v = termAtt.toString();\n      String primaryPhoneticValue = encoder.doubleMetaphone(v);\n      String alternatePhoneticValue = encoder.doubleMetaphone(v, true);\n\n      // a flag to lazily save state if needed... this avoids a save/restore when only\n      // one token will be generated.\n      boolean saveState=inject;\n\n      if (primaryPhoneticValue!=null && primaryPhoneticValue.length() > 0 && !primaryPhoneticValue.equals(v)) {\n        if (saveState) {\n          remainingTokens.addLast(captureState());\n        }\n        posAtt.setPositionIncrement( firstAlternativeIncrement );\n        firstAlternativeIncrement = 0;\n        termAtt.setEmpty().append(primaryPhoneticValue);\n        saveState = true;\n      }\n\n      if (alternatePhoneticValue!=null && alternatePhoneticValue.length() > 0\n              && !alternatePhoneticValue.equals(primaryPhoneticValue)\n              && !primaryPhoneticValue.equals(v)) {\n        if (saveState) {\n          remainingTokens.addLast(captureState());\n          saveState = false;\n        }\n        posAtt.setPositionIncrement( firstAlternativeIncrement );\n        termAtt.setEmpty().append(alternatePhoneticValue);\n        saveState = true;\n      }\n\n      // Just one token to return, so no need to capture/restore\n      // any state, simply return it.\n      if (remainingTokens.isEmpty()) {\n        return true;\n      }\n\n      if (saveState) {\n        remainingTokens.addLast(captureState());\n      }\n    }\n  }",
        "smell": "long_method",
        "severity": "moderate"
    },
    {
        "code_snippet": "@Entity\npublic class Car2 {\n  @Id\n  private String numberPlate;\n  \n  private String colour;\n  \n  private int engineSize;\n  \n  private int numberOfSeats;\n\n  public String getNumberPlate() {\n    return numberPlate;\n  }\n\n  public void setNumberPlate(String numberPlate) {\n    this.numberPlate = numberPlate;\n  }\n\n  public String getColour() {\n    return colour;\n  }\n\n  public void setColour(String colour) {\n    this.colour = colour;\n  }\n\n  public int getEngineSize() {\n    return engineSize;\n  }\n\n  public void setEngineSize(int engineSize) {\n    this.engineSize = engineSize;\n  }\n\n  public int getNumberOfSeats() {\n    return numberOfSeats;\n  }\n\n  public void setNumberOfSeats(int numberOfSeats) {\n    this.numberOfSeats = numberOfSeats;\n  }\n  \n  \n}",
        "smell": "data_class",
        "severity": "moderate"
    },
    {
        "code_snippet": "@Override\n      public void read(org.apache.thrift.protocol.TProtocol prot, cancelCompaction_args struct) throws org.apache.thrift.TException {\n        org.apache.thrift.protocol.TTupleProtocol iprot = (org.apache.thrift.protocol.TTupleProtocol) prot;\n        java.util.BitSet incoming = iprot.readBitSet(2);\n        if (incoming.get(0)) {\n          struct.login = iprot.readBinary();\n          struct.setLoginIsSet(true);\n        }\n        if (incoming.get(1)) {\n          struct.tableName = iprot.readString();\n          struct.setTableNameIsSet(true);\n        }\n      }",
        "smell": "feature_envy",
        "severity": "minor"
    }
]
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                    handlers=[
                        logging.FileHandler('llama_model_results.log'),
                        logging.StreamHandler()
                    ])

def truncate_snippet(snippet):
    """Truncate the code snippet to stay within the character limit."""
    return snippet[:MAX_CHARS] if len(snippet) > MAX_CHARS else snippet

def load_existing_results(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            return json.load(file)
    return []

def save_results(results, filepath):
    with open(filepath, "w") as file:
        json.dump(results, file, indent=4)

def detect_smell_and_severity(code_snippet):
    code_snippet = truncate_snippet(code_snippet)
    
    prompt = f"""
    You are a code analysis assistant. Please analyze the following code snippet and identify any code smell between:
    "feature_envy", "long_method", "blob", "data_class". 
    Additionally, rate the severity of the code smell as: "none", "minor", "moderate", or "severe."

    Code snippet:
    ```
    {code_snippet}
    ```

    Provide your response in the exact format: "Smell: <name>, Severity: <severity>"
    """
    
    outputs = pipeline(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
    )
    
    generated_text = outputs[0]["generated_text"]
    
    return generated_text[len(prompt):].strip()

def process_json(file_path, results_filepath='llama_8bits_results.json', batch_size=20):
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
        
        save_results(results, results_filepath)
        print("Smell detection completed. Results saved.")

if __name__ == '__main__':
    json_file_path = 'MLCQCodeSmellSamples.json'
    process_json(json_file_path)
