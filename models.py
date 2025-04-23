import numpy as np
import pandas as pd
from typing import Dict, Callable, Any, Tuple, List
from tqdm import tqdm

class Proxy():
    def __init__(
        self,
        proxy_inputs: Dict[str, str],
        proxy_func:Callable[[str], Tuple[Any, float]],
        verbose:bool=True
    ):
        self.inputs = proxy_inputs
        self.proxy_func = proxy_func
        self.preds_dict={}
        self.verbose=verbose


    def get_preds_and_scores(self, indxs) -> Tuple[np.ndarray, np.ndarray]:
        preds = []
        scores = []
        for x in tqdm(indxs, disable=not self.verbose):
            if x in self.preds_dict:
                pred, score = self.preds_dict[x]
            else:
                pred, score = self.proxy_func(self.inputs[x])
                self.preds_dict[x] = (pred, score)
            preds.append(pred)
            scores.append(score)
            
        return np.array(scores), np.array(preds)


class Oracle():
    def __init__(
        self,
        oracle_inputs: Dict[str, str],
        oracle_func: Callable[[str, str], Tuple[bool, Any]],
        verbose:bool=True
    ):
        self.inputs = oracle_inputs
        self.oracle_func = oracle_func

        self.cached_validations={}
        self.preds_dict={}
        self.verbose=verbose

    def get_pred(self, indxs:List) -> np.ndarray:
        preds = []
        for x in tqdm(indxs, disable=not self.verbose):
            if x in self.preds_dict:
                oracle_output = self.preds_dict[x]
            else:
                _, oracle_output = self.oracle_func(self.inputs[x], "")
            self.preds_dict[x] = oracle_output
            preds.append(oracle_output)
        return np.array(preds)


    def get_number_preds(self) -> int:
        return len(self.preds_dict)


    def is_answer_correct(self, data_indxs:List, proxy_output_at_indxs:List) -> np.ndarray:
        validations = []
        for i, x in tqdm(enumerate(data_indxs), disable=not self.verbose):
            proxy_output =proxy_output_at_indxs[i] 
            if (x, proxy_output) in self.cached_validations:
                is_correct = self.cached_validations[(x, proxy_output)]
            else:
                is_correct, oracle_output = self.oracle_func(self.inputs[x], proxy_output)
                self.preds_dict[x] = oracle_output
                self.cached_validations[(x, proxy_output)] = is_correct
            validations.append(is_correct)

        return np.array(validations)


class PrecomputedOracle(Oracle):
    def __init__(
        self,
        indexes,
        preds,
        verbose=True
    ):
        preds_dict = {'ids':indexes, 'preds':preds}
        preds_df = pd.DataFrame.from_dict(preds_dict).set_index('ids')
        def oracle_func(data_indx, proxy_output, df=preds_df):
            oracle_output = df.loc[data_indx, 'preds']
            return oracle_output == proxy_output, oracle_output


        super().__init__(oracle_func=oracle_func, oracle_inputs={key: key for key in indexes}, verbose=verbose)

class PrecomputedProxy(Proxy):
    def __init__(
        self,
        indexes,
        preds,
        scores,
        verbose=True
    ):
        preds_dict = {'ids':indexes, 'preds':preds, 'scores':scores}
        preds_df = pd.DataFrame.from_dict(preds_dict).set_index('ids')
        def proxy_func(data_indx, df=preds_df):
            proxy_output = df.loc[data_indx, 'preds']
            proxy_score = df.loc[data_indx, 'scores']
            return proxy_score, proxy_output

        super().__init__(proxy_func=proxy_func, proxy_inputs={key: key for key in indexes}, verbose=verbose)
