
import numpy as np
import pandas as pd

from PRISM import PRISM_A, GPT4omini, GPT4o

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
df['id'] = np.arange(len(df))

# Define oracle and proxy
oracle = GPT4o(df['id'].to_numpy(), df['opinion_text'].to_numpy(), task, is_binary=True)
proxy = GPT4omini(df['id'].to_numpy(), df['opinion_text'].to_numpy(), task, is_binary=True)

# Call PRISM to process
prism = PRISM_A(df['id'].to_numpy(), proxy, oracle, 0.1, 0.9, seed=0)
output_df = prism.process()

# Evaluate output 
print('Running oracle for evaluation')
output_df["used_oracle"] = output_df["used_oracle"].astype(bool)
proxy_output_df = output_df[~output_df['used_oracle']].reset_index()
used_proxy_count = len(proxy_output_df)
indxs = proxy_output_df['data_indx'].to_numpy()
oracle_outputs = oracle.get_pred(indxs)
df_new_oracle = pd.DataFrame.from_dict({'data_indx':indxs, 'output':oracle_outputs, 'used_oracle':[True]*len(indxs)})
df_all_oracle = pd.concat([df_new_oracle, output_df[output_df['used_oracle']].reset_index()]).set_index('data_indx')
df_joined = df_all_oracle.join(output_df, rsuffix='_including_proxy')
df_joined['is_correct'] = df_joined['output'] == df_joined['output_including_proxy']
print(f"Accuracy: {df_joined['is_correct'].mean()}, Used Proxy: {used_proxy_count}")
df_joined.to_csv('res_court_opinion.csv')