import numpy as np
import pandas as pd

from BARGAIN import OpenAIProxy, OpenAIOracle
from BARGAIN import BARGAIN_A

# Define Data and Task
task = ''' 
I will give you a Supreme Court opinion.
Your task is to determine if this opinion reverses a lower court's ruling.
Note that the opinion may not be an appeal, but rather a new ruling.

- True if the Supreme Court reverses the lower court ruling
- False otherwise

Here is the opinion: {}

You must respond with ONLY True or False:
'''
df = pd.read_csv(f'court_opinion.csv')

# Define oracle and proxy
proxy = OpenAIProxy(task, model='gpt-4o-mini', is_binary=True)
oracle = OpenAIOracle(task, model='gpt-4o', is_binary=True)

# Call BARGAIN to process
bargain = BARGAIN_A(proxy, oracle, target=0.9, delta=0.1, seed=0)
output, used_oracle = bargain.process(df['opinion_text'].to_numpy(), return_oracle_usage=True)
df['output'] = output
df['used_oracle'] = used_oracle
used_proxy_count = 1-df['used_oracle'].mean()

# Evaluate output 
print('Running oracle on all records for evaluation')
df['output_oracle']=df['output']
df.loc[~df['used_oracle'], 'output_oracle'] = oracle.get_pred(df.loc[~df['used_oracle'], 'opinion_text'].to_numpy())
df['is_correct'] = df['output_oracle'] == df['output']
print(f"Accuracy: {df['is_correct'].mean()}, Used Proxy: {used_proxy_count:.2f}")
