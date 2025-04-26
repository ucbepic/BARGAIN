# PRISM: Low-Cost LLM-Powered Data Processing with Guarantees
PRISM helps reduce cost when processing a dataset using LLMs. It automatically decides when to use a cheap and potentially inaccurate LLM, or an expensive but accurate LLM when processing the data, while providing accuracy guarantees. It maximizes how often the cheap LLM is used while guaranteeing the answer matches the expensive LLM's output based on a user-provided accuracy requirement.  


## Overview
PRISM follows the _model cascade_ framework. To process a data record with an LLM given a prompt, it first runs the cheap model on the data record. Based on the model's output logprobabilities it decides whether to trust the cheap model or not. If it decides the cheap model's output is inaccurate, it then runs the more expensive model. 

<p align="center">
<img src="https://github.com/szeighami/PRISM/blob/main/images/PRISM_workflow.png" width="500">
</p>

To decide whether to trust the cheap model, PRISM uses a _cascade threshold_: if the cheap model's output logprobability is more than the cascade threshold, PRISM uses the cheap model's output. This cascade threshold is determined in a preprocessing step to provide theoretical guarantees. Given an accuracy target `T`, **PRISM guarantees the output matches the expensive model's output at least on T% of the documents, but uses the cheap model on as many records as possible.** This guarantee is achieved through sampling and labeling a few records during preprocessing to estimate a suitable cascade threshold. 

## Installation
To install PRISM, run
```bash
pip install ai-prism
```
PRISM uses `numpy`, `pandas`, `tqdm`, and `openai` libraries. The `openai` library is optional and can be replaced with other service providers.

## Getting Started
Assume you have a dataset you want to process using LLMs with a specific prompt. We consider a toy example here:
```python
data_records= ['zebra', 'monkey', 'red', 'blue', 'lion', 'black']
task= "Does the text '{}' mention an animal?"
```
`data` is a list of strings and `task` is a templatized string. Our goal is to prompt an LLM with `task.format(data_records[i])` for all `i` to obtain whether each data item is an animal or a color.

We do so using PRISM and OpenAI models. OpenAI provides `gpt-4o` and `gpt-4o-mini` for processing. PRISM automatically decides which one to use, while guaranteeing the output matches gpt-4o based on a user-provided accuracy requirement. 

To use PRISM, first import
```python
from PRISM import OpenAIProxy, OpenAIOracle PRISM_A
```
PRISM refers to the cheap but potentially inaccurate model as _proxy_ and to the expensive but accurate model as _oracle_. Here, for OpenAI models, `gpt-4o` is our oracle and `gpt-4o-mini` is our proxy. We first define them below
```python
proxy = OpenAIProxy(task, model='gpt-4o-mini', is_binary=True)
oracle = OpenAIOracle(task, model='gpt-4o', is_binary=True)
```
`task` is the templatized string defined above, `model` is the name of the model to use and `is_binary` denotes whether the task is a binary classification task (as is in our case). You can use PRISM for non-binary classification or open-ended tasks as well, see HERE for an example.   

Then, to use PRISM, run:
```python
prism = PRISM_A(proxy, oracle, target=0.9, delta=0.1)
res = prism.process(data)
```
`PRISM_A` is the main class used for processing (`A` stands for accuracy, see here when considering precision or recall metrics). `target` is the accuracy requirement, `target=0.9` means 90% of outputs must match those of the oracle (`gpt-4o` in this example). `delta` is a probability of failure. Our guarantees are statistical, and `delta` specifies the probability that the guarantee may not hold. For example, `target=0.9, delta=0.1` means at least 90% of outputs must match the oracle's outputs at least `1-delta=0.9` percent of the time. 

Calling `prism.process(data)` processes the data and returns a list, with `len(res)=len(data)` and `res` contains the LLM output for each data record. 

## Examples
[examples](https://github.com/szeighami/PRISM/tree/main/examples) folder contains multiple example use-cases. _Run examples from the examples directory_.

**To run the examples, you must set your OpenAI API key. ** As of this writing, the Color or Animal and Extract Animal examples cost less than 1$, and the Supreme Court Opinion example costs about 10$.
### Color or Animal
This is an extension of the toy example discussed above. Run
```bash
python toy_dataset_color_or_animal.py
```
The example generates a synthetic dataset containing color or animal names, and asks the LLMs to decide whether there a record mentions an animal or not. You will get the output`
```
Accuracy: 0.95, Used Proxy: 0.45
```
This means PRISM used the proxy (i.e., `gpt-4o-mini`) to process 45% of the records, but the output matches the oracle's output (i.e., `gpt-4o`) on 95% of the records. 

### Supreme Court Opinion
This is an example on a real-world dataset, obtained from https://www.courtlistener.com/. Each record in [the dataset](https://github.com/szeighami/PRISM/blob/main/examples/court_opinion.csv) consists of a Supreme Court written opinion, and the task is to determine whether the opinion reverses a lower court ruling. Run 
```bash
python court_opinion_example.py
```
We obtain
```
Accuracy: 0.976, Used Proxy: 0.406
```
This means PRISM used the proxy (i.e., `gpt-4o-mini`) to process 40.6% of the records, but the output matches the oracle's output (i.e., `gpt-4o`) on 97.6% of the records. 

### Extract Animal
This example uses PRISM for an open-ended task. It generates a dataset where each record consists of a description of color theory, but an animal name is inserted in the middle of the text. The task for the LLM is to extract the animal name. Run
```bash
python toy_dataset_extract_animal.py
```
We obtain
```
Accuracy: 1.0, Used Proxy: 0.57
```
This means PRISM used the proxy (i.e., `gpt-4o-mini`) to process 57% of the records, but the output matches the oracle's output (i.e., `gpt-4o`) on 100% of the records. 


## Using Other Models

## Precision and Recall Targets

