
import numpy as np

from PRISM import GPT4omini, GPT4o
from PRISM import PRISM_A
from generate_toy_data import generate_color_or_animal_data

# Define Data and Task
df = generate_color_or_animal_data(n=1000, animal_prop=0.5, hard_prop=0.5, misleading_text_length=1000)
task = ''' 
        I will give you a text. Your task is to determine if the text mentions an animal.

        - True if it mentions an animal
        - False otherwise

        Here is the text: {}

        You must respond with ONLY True or False:
        '''

# Define oracle and proxy
oracle = GPT4o(df['id'].to_numpy(), df['value'].to_numpy(), task, is_binary=True)
proxy = GPT4omini(df['id'].to_numpy(), df['value'].to_numpy(), task, is_binary=True)

# Call PRISM to process
prism = PRISM_A(df['id'].to_numpy(), proxy, oracle, 0.1, 0.9, seed=0)
output_df = prism.process()

# Evaluate output 
df_joined = df.set_index('id').join(output_df)
df_joined['is_correct'] = df_joined['is_animal'] == df_joined['output']
print(f"Accuracy: {df_joined['is_correct'].mean()}, Used Proxy: {1-oracle.get_number_preds()/len(df)}")