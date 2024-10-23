import json
import logging
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in file: {file_path}")
        return None

def normalize_smell(smell):
    smell = smell.lower().replace('_', ' ')
    if 'data class' in smell:
        return 'data class'
    elif 'long method' in smell:
        return 'long method'
    elif 'feature envy' in smell:
        return 'feature envy'
    elif 'blob' in smell:
        return 'blob'
    else:
        return 'unknown'

def normalize_severity(severity):

    severity = severity.lower().strip()
    return severity

def parse_result(result):

    smells = []
    result = result.lower().replace('"', '')
    lines = result.split('\n')
    
    for line in lines:
        smell = severity = None
        if 'smell:' in line and 'severity:' in line:
            parts = line.split(',')
            for part in parts:
                if 'smell:' in part:
                    smell = normalize_smell(part.split(':')[1].strip())
                elif 'severity:' in part:
                    severity = normalize_severity(part.split(':')[1].strip())
            if smell and severity:
                smells.append((smell, severity))
    
    return smells

def compare_results(model_output, ground_truth):
    model_smells = parse_result(model_output)
    true_smell = normalize_smell(ground_truth['smell'])
    true_severity = normalize_severity(ground_truth['severity'])

    if not model_smells:
        logging.warning(f"Invalid format in model output: {model_output}")
        return False

    if true_severity == 'none' and all(severity == 'none' for _, severity in model_smells):
        return True

    return any(smell == true_smell and severity == true_severity for smell, severity in model_smells)

def compute_metrics(results, ground_truth):
    y_true = []
    y_pred = []
    smells = ['data class', 'long method', 'feature envy', 'blob']

    for result in results:
        unique_id = result['unique_id']
        if unique_id not in ground_truth:
            logging.warning(f"Unique ID {unique_id} not found in ground truth data")
            continue

        model_output = result['smell_and_severity']
        truth = ground_truth[unique_id]
        
        true_smell = normalize_smell(truth['smell'])
        true_severity = normalize_severity(truth['severity'])
        
        y_true.append(true_smell if true_severity != 'none' else 'none')
        
        model_smells = parse_result(model_output)
        predicted_smell = 'none'
        for smell, severity in model_smells:
            if severity != 'none' and smell in smells:
                predicted_smell = smell
                break
        
        y_pred.append(predicted_smell)

    return y_true, y_pred

def main():
    results_filepath = 'gpt4_results.json'
    origin_filepath = 'MLCQCodeSmellSamples.json'
    results = load_json(results_filepath)
    ground_truth_data = load_json(origin_filepath)

    if results is None or ground_truth_data is None:
        logging.error("Failed to load necessary data. Exiting.")
        return

    logging.info(f"Loaded {len(results)} results and {len(ground_truth_data)} ground truth items")

    ground_truth = {item['unique_id']: {'smell': item['smell'], 'severity': item['severity']} for item in ground_truth_data}

    y_true, y_pred = compute_metrics(results, ground_truth)
    
    smells = ['data class', 'long method', 'feature envy', 'blob']
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=smells, average=None)

    for i, smell in enumerate(smells):
        logging.info(f"Metrics for {smell}:")
        logging.info(f"  Precision: {precision[i]:.4f}")
        logging.info(f"  Recall: {recall[i]:.4f}")
        logging.info(f"  F1 Score: {f1[i]:.4f}")

if __name__ == "__main__":
    main()
