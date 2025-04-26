import numpy as np
from typing import Tuple, List, Any
from tqdm import tqdm

class Proxy():
    def __init__(
        self,
        verbose:bool=True
    ):
        self.preds_dict={}
        self.verbose=verbose

    def proxy_func(self, input: str) -> Tuple[Any, float]:
        '''
        Must extend Proxy class and specifiy this function. This function processes `input` with Proxy. 
        It returns a tuple, with the first element denoting the output of proxy, and the second element the proxy score
    
        Args:
            input: Data record to be processed by the oracle

        Returns:
            Tuple[Any, float]:
                -- Any: The output for `input` computed by proxy model
                -- float: The proxy score computed for the model
        '''
        assert False << "SUBCLASS MUST IMPLEMENT"

    def reset(self) -> None:
        self.preds_dict={}

    def get_preds_and_scores(self, indxs:List, data_records:List) -> Tuple[np.ndarray, np.ndarray]:
        preds = []
        scores = []
        tqdm_bar = len(indxs)>20 and self.verbose
        for i, x in tqdm(enumerate(indxs), disable=(not tqdm_bar), total=len(indxs)):
            if x in self.preds_dict:
                pred, score = self.preds_dict[x]
            else:
                pred, score = self.proxy_func(data_records[i])
                self.preds_dict[x] = (pred, score)
            preds.append(pred)
            scores.append(score)
            
        return np.array(preds), np.array(scores)


class Oracle():
    def __init__(
        self,
        verbose:bool=True
    ):
        self.cached_validations={}
        self.preds_dict={}
        self.verbose=verbose

    def get_pred(self, data_records:List, indxs:List=None) -> np.ndarray:
        preds = []
        tqdm_bar = len(data_records)>20 and self.verbose
        for i, record in tqdm(enumerate(data_records), disable=(not tqdm_bar), total=len(data_records)):
            if indxs is not None and indxs[i] in self.preds_dict:
                oracle_output = self.preds_dict[indxs[i]]
            else:
                _, oracle_output = self.oracle_func(record, "")
            if indxs is not None:
                self.preds_dict[indxs[i]] = oracle_output
            preds.append(oracle_output)
        return np.array(preds)

    def oracle_func(self, input: str, proxy_output: Any) -> Tuple[bool, Any]:
        '''
        Must extend Oracle class and specifiy this function. This function checks if a given `proxy_output` is correct for a given `input`. 
        It returns a tuple, with the first element denoting whether the `proxy_output` is correct, and the second element denotes the correct answer for `input`
    
        Args:
            input: Data record to be processed by the oracle
            proxy_output: Output provided by the Proxy on `input`. Oracle needs to validate if `proxy_output` is correct

        Returns:
            Tuple[bool, Any]:
                -- bool: Whether `proxy_output` is correct for `input` 
                -- Any: The correct output for `input` (can be the same as `proxy_output`)
        '''
        assert False << "MUST IMPLEMENT"


    def get_number_preds(self) -> int:
        return len(self.preds_dict)

    def reset(self) -> None:
        self.cached_validations={}
        self.preds_dict={}

    def is_answer_correct(self, data_indxs:List, data_records:List, proxy_output_at_indxs:List) -> np.ndarray:
        validations = []
        tqdm_bar = len(data_indxs)>20 and self.verbose
        for i, x in tqdm(enumerate(data_indxs), disable=not tqdm_bar, total=len(data_indxs)):
            proxy_output =proxy_output_at_indxs[i] 
            if (x, proxy_output) in self.cached_validations:
                is_correct = self.cached_validations[(x, proxy_output)]
            else:
                is_correct, oracle_output = self.oracle_func(data_records[i], proxy_output)
                self.preds_dict[x] = oracle_output
                self.cached_validations[(x, proxy_output)] = is_correct
            validations.append(is_correct)

        return np.array(validations)
