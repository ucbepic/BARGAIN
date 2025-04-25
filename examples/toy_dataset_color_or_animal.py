from PRISM import OpenAIProxy, OpenAIOracle
from PRISM import PRISM_A
from generate_toy_data import generate_color_or_animal_data

# Define Data and Task
df = generate_color_or_animal_data(n=100, animal_prop=0.5, hard_prop=0.5, misleading_text_length=600)
task = ''' 
        I will give you a text. Your task is to determine if the text mentions an animal.

        - True if it mentions an animal
        - False otherwise

        Here is the text: {}

        You must respond with ONLY True or False:
        '''

# Define oracle and proxy
proxy = OpenAIProxy(task, model='gpt-4o-mini', is_binary=True)
oracle = OpenAIOracle(task, model='gpt-4o', is_binary=True)

# Call PRISM to process
prism = PRISM_A(proxy, oracle, target=0.9, delta=0.1, seed=0)
df['output'] = prism.process(df['value'].to_numpy())

# Evaluate output 
df['is_correct'] = df['is_animal'] == df['output']
print(f"Accuracy: {df['is_correct'].mean()}, Used Proxy: {1-oracle.get_number_preds()/len(df):.2}")