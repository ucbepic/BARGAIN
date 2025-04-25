from openai import OpenAI
import os
import json
from pydantic import BaseModel
import numpy as np

from PRISM.models.AbstractModels import Oracle, Proxy

class GeneralOracleAnswer(BaseModel):
  is_correct: bool
  correct_answer: str

def get_bool_val_prob(res, logprobs=None):
    if logprobs is None:
        output=False
        if 'true' in res.lower() and 'false' not in res.lower():
            output=True
        return output
    
    true_prob = 0
    false_prob = 0
    for toplogprob in logprobs[0].top_logprobs:
        if toplogprob.token == 'True':
            true_prob = np.exp(toplogprob.logprob)
        if toplogprob.token == 'False':
            false_prob = np.exp(toplogprob.logprob)
    if true_prob == 0 and false_prob == 0:
        return False, 0
    norm = true_prob+false_prob
    true_prob = true_prob/norm
    false_prob = false_prob/norm
    if true_prob>false_prob:
        return True, true_prob 
    return False, false_prob




class GPT4omini(Proxy):
    def __init__(
                self,
                indexes:np.ndarray,
                data:np.ndarray,
                task:str,
                is_binary:bool=False,
                verbose:bool=True
            ) -> None :
        '''
        Args: 
            indexes: Identifies for each data record. The identifier for a record is passed to the model when processing the record. 
            data: list of data inputs indexes by `indexes`. For the `i`-th index, `data[i]` contains the data for record with identifier `indexes[i]`
            task: prompt to perform on each item in the `data` list. `task` must be a templatized string: `task.format(data[i])` is passed to gpt4o-mini to process the `i`-th data record
            is_binary: Set to `True` if the task is a binary classifiction task
            verbose: provide progress updates

        '''
        super().__init__(proxy_inputs={key: data[i] for i, key in enumerate(indexes)}, verbose=verbose)
        self.task = task
        self.is_binary=is_binary

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def proxy_func_general(self, data_record):
        task_with_data = self.task.format(data_record)
        prompt=[
                    {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
                    {"role": "user", "content": task_with_data}
                    ]
        response = self.client.beta.chat.completions.parse( model='gpt-4o-mini', messages=prompt, logprobs=True, seed=0, temperature=0)
        if response.choices[0].logprobs is None:
            prob = 0
        else:
            logprobs = response.choices[0].logprobs.content
            all_logprobs = 0
            for t in logprobs:
                all_logprobs+= t.logprob
            prob = np.exp(all_logprobs)

        return response.choices[0].message.content, prob

    def proxy_func_binary(self, data_record):
        task_with_data = self.task.format(data_record)
        prompt=[
                    {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
                    {"role": "user", "content": task_with_data}
                    ]
        response = self.client.beta.chat.completions.parse( model='gpt-4o-mini', messages=prompt, logprobs=True, seed=0, temperature=0, max_tokens=2, top_logprobs=10)
        res =response.choices[0].message.content
        logprobs = response.choices[0].logprobs.content
        return get_bool_val_prob(res, logprobs)

    def proxy_func(self, data_record):
        if self.is_binary:
            return self.proxy_func_binary(data_record)
        else:
            return self.proxy_func_general(data_record)

class GPT4o(Oracle):
    def __init__(
        self,
        indexes:np.ndarray,
        data:np.ndarray,
        task:str,
        is_binary:bool=False,
        verbose:bool=True
    ):
        '''
        Args: 
            indexes: Identifies for each data record. The identifier for a record is passed to the model when processing the record. 
            data: list of data inputs indexes by `indexes`. For the `i`-th index, `data[i]` contains the data for record with identifier `indexes[i]`
            task: prompt to perform on each item in the `data` list. `task` must be a templatized string: `task.format(data[i])` is passed to gpt4o to process the `i`-th data record
            is_binary: Set to `True` if the task is a binary classifiction task
            verbose: provide progress updates

        '''
        super().__init__(oracle_inputs={key: data[i] for i, key in enumerate(indexes)}, verbose=verbose)
        self.task = task
        self.is_binary=is_binary

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def oracle_func_binary(self, data_record, proxy_output):
        task_with_data = self.task.format(data_record)
        prompt=[
                    {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
                    {"role": "user", "content": task_with_data}
                ]
        response = self.client.beta.chat.completions.parse( model='gpt-4o', messages=prompt, logprobs=False, seed=0, temperature=0, max_tokens=2)
        res=response.choices[0].message.content
        oracle_output = get_bool_val_prob(res)
        return oracle_output == proxy_output, oracle_output

    def oracle_func_general(self, data_record, proxy_output):
        task_with_data = self.task.format(data_record)
        prompt=[
                    {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
                    {"role": "user", "content": f'''
                        Consider the following task and a given response:
                        
                        Task:
                        {task_with_data}

                        Response: {proxy_output}

                        Is the provided response correct? If the provided answer is incorrect, provide the correct answer.
                        '''}
                ]
        response = self.client.beta.chat.completions.parse( model='gpt-4o', messages=prompt, response_format=GeneralOracleAnswer, logprobs=False, seed=0, temperature=0)
        res=json.loads(response.choices[0].message.content)
        correct_answer = res['correct_answer']
        if res['is_correct']:
            correct_answer = proxy_output
        return res['is_correct'], correct_answer
    
    def oracle_func(self, data_record, proxy_output):
        if self.is_binary:
            return self.oracle_func_binary(data_record, proxy_output)
        else:
            return self.oracle_func_general(data_record, proxy_output)