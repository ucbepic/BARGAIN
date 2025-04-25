
import numpy as np
import pandas as pd

from PRISM import GPT4omini, GPT4o
from PRISM import PRISM_P, PRISM_R
from generate_toy_data import generate_color_or_animal_data

def precision(indexes, labels, pred_indexes):
    if len(pred_indexes) == 0:
        return 1
    df = pd.DataFrame.from_dict({'indexes':indexes, 'labels':labels}).set_index('indexes')
    return df.loc[pred_indexes, 'labels'].mean()

def recall(indexes, labels, pred_indexes):
    df = pd.DataFrame.from_dict({'indexes':indexes, 'labels':labels}).set_index('indexes')
    no_pos =(df['labels']==1).sum()
    no_pos_pred = df.loc[pred_indexes, 'labels'].sum()
    return no_pos_pred/no_pos





def test_given_precision_requirement(data_df, task, k, target, delta):
    # Define oracle and proxy
    oracle = GPT4o(data_df['id'].to_numpy(), data_df['value'].to_numpy(), task, is_binary=True)
    proxy = GPT4omini(data_df['id'].to_numpy(), data_df['value'].to_numpy(), task, is_binary=True)

    # Call PRISM to process
    prism = PRISM_P(data_df['id'].to_numpy(), proxy, oracle, delta, target, k, seed=0)
    est_positive_indx = prism.process()

    # Evalute
    est_precision = precision(data_df['id'].to_numpy(), data_df['is_animal'].to_numpy(), est_positive_indx)
    est_recall = recall(data_df['id'].to_numpy(), data_df['is_animal'].to_numpy(), est_positive_indx)
    print(f"Returned a set with Precision: {est_precision}, Recall: {est_recall} using {k} oracle calls given precision target {target*100}%")


def test_given_recall_requirement(data_df, task, k, target, delta):
    # Define oracle and proxy
    oracle = GPT4o(data_df['id'].to_numpy(), data_df['value'].to_numpy(), task, is_binary=True)
    proxy = GPT4omini(data_df['id'].to_numpy(), data_df['value'].to_numpy(), task, is_binary=True)

    # Call PRISM to process
    prism = PRISM_R(data_df['id'].to_numpy(), proxy, oracle, delta, target, k, seed=0)
    est_positive_indx = prism.process()

    # Evalute
    est_precision = precision(data_df['id'].to_numpy(), data_df['is_animal'].to_numpy(), est_positive_indx)
    est_recall = recall(data_df['id'].to_numpy(), data_df['is_animal'].to_numpy(), est_positive_indx)
    print(f"Returned a set with Precision: {est_precision}, Recall: {est_recall} using {k} oracle calls given recall target {target*100}%")




# Define Data and Task
df = generate_color_or_animal_data(n=500, animal_prop=0.5, hard_prop=0.5, misleading_text_length=1000)
task = ''' 
        I will give you a text. Your task is to determine if the text mentions an animal.

        - True if it mentions an animal
        - False otherwise

        Here is the text: {}

        You must respond with ONLY True or False:
        '''
k=100
target = 0.9
delta=0.1
test_given_precision_requirement(df, task, k, target, delta)
test_given_recall_requirement(df, task, k, target, delta)
