import pandas as pd

from PRISM.models.AbstractModels import Oracle, Proxy


class PrecomputedOracle(Oracle):
    def __init__(
        self,
        indexes,
        preds,
        verbose=False
    ):
        super().__init__(oracle_inputs={key: key for key in indexes}, verbose=verbose)
        preds_dict = {'ids':indexes, 'preds':preds}
        self.preds_df = pd.DataFrame.from_dict(preds_dict).set_index('ids')

    def oracle_func(self, input, proxy_output):
        oracle_output = self.preds_df.loc[input, 'preds']
        return oracle_output == proxy_output, oracle_output


class PrecomputedProxy(Proxy):
    def __init__(
        self,
        indexes,
        preds,
        scores,
        verbose=False
    ):
        super().__init__(proxy_inputs={key: key for key in indexes}, verbose=verbose)

        preds_dict = {'ids':indexes, 'preds':preds, 'scores':scores}
        self.preds_df = pd.DataFrame.from_dict(preds_dict).set_index('ids')


    def proxy_func(self, input):
        proxy_output = self.preds_df.loc[input, 'preds']
        proxy_score = self.preds_df.loc[input, 'scores']
        return proxy_output, proxy_score