# PRISM: Low-Cost LLM-Powered Data Processing
PRISM helps reduce cost when processing a dataset using LLMs. It automatically decides when to use a cheap and potentially inaccurate LLM, or an expensive but accurate LLM when processing the data, while providing accuracy guarantees. It maximizes how often the cheap LLM is used while guaranteeing the answer matches the expensive LLM's output based on a user-provided accuracy requirement.  

<p align="center">
<img src="https://github.com/szeighami/nudge/blob/main/PRISM_workflow.png" width="500">
</p>


## Getting Started
To install PRISM, run
```bash
pip install ai-prism
```
