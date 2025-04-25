import numpy as np
from typing import Dict, Tuple, List
from tqdm import tqdm

class Proxy():
    def __init__(
        self,
        proxy_inputs: Dict[str, str],
        verbose:bool=True
    ):
        self.inputs = proxy_inputs
        self.preds_dict={}
        self.verbose=verbose

    def proxy_func(self, input):
        assert False << "SUBCLASS MUST IMPLEMENT"

    def get_preds_and_scores(self, indxs) -> Tuple[np.ndarray, np.ndarray]:
        preds = []
        scores = []
        tqdm_bar = len(indxs)>20 and self.verbose
        for x in tqdm(indxs, disable=(not tqdm_bar)):
            if x in self.preds_dict:
                pred, score = self.preds_dict[x]
            else:
                pred, score = self.proxy_func(self.inputs[x])
                self.preds_dict[x] = (pred, score)
            preds.append(pred)
            scores.append(score)
            
        return np.array(preds), np.array(scores)


class Oracle():
    def __init__(
        self,
        oracle_inputs: Dict[str, str],
        verbose:bool=True
    ):
        self.inputs = oracle_inputs

        self.cached_validations={}
        self.preds_dict={}
        self.verbose=verbose

    def get_pred(self, indxs:List) -> np.ndarray:
        preds = []
        tqdm_bar = len(indxs)>20 and self.verbose
        for x in tqdm(indxs, disable=True):#not tqdm_bar):
            if x in self.preds_dict:
                oracle_output = self.preds_dict[x]
            else:
                _, oracle_output = self.oracle_func(self.inputs[x], "")
            self.preds_dict[x] = oracle_output
            preds.append(oracle_output)
        return np.array(preds)

    def oracle_func(self, input, proxy_output):
        assert False << "MUST IMPLEMENT"


    def get_number_preds(self) -> int:
        return len(self.preds_dict)


    def is_answer_correct(self, data_indxs:List, proxy_output_at_indxs:List) -> np.ndarray:
        validations = []
        tqdm_bar = len(data_indxs)>20 and self.verbose
        for i, x in tqdm(enumerate(data_indxs), disable=True):#not tqdm_bar):
            proxy_output =proxy_output_at_indxs[i] 
            if (x, proxy_output) in self.cached_validations:
                is_correct = self.cached_validations[(x, proxy_output)]
            else:
                is_correct, oracle_output = self.oracle_func(self.inputs[x], proxy_output)
                self.preds_dict[x] = oracle_output
                self.cached_validations[(x, proxy_output)] = is_correct
            validations.append(is_correct)

        return np.array(validations)
