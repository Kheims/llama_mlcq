# Reproducing Code Smell Detection Results


First, create a Python environment and install the required dependencies.

```bash
conda create -n venv python=3.10

conda activate venv

pip install -r requirements.txt

```
Set your github token to download the code snippets associated with the dataset through the DataExtractor

```
export GITHUB_TOKEN=<your_github_token>

python DataExtracor.py
```

Then run the gpt script :
```
python gpt4.py
```

If you have a Cuda capable setup run the llama script :

```
python llama.py
```

Finally to compute the metrics, run :

```
python compute_metrics.py
```
