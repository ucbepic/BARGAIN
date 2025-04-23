import numpy as np

class WoR_Sampler():
    def __init__(
        self,
        n,
    ):
        self.indxs  = np.random.permutation(n)
        self.sampled_at_threshs  = {}
        self.all_sampled = np.array([])


    def sample(self, thresh, k):
        if  thresh not in self.sampled_at_threshs:
            self.sampled_at_threshs[thresh] = 0
        
        t = self.sampled_at_threshs[thresh]
        curr_indxs = self.indxs[self.indxs <=thresh]
        if t >= len(curr_indxs):
            return np.array([])
        sample = curr_indxs[t:t+k]
        t+=len(sample)
        sampled_all = False
        if t >= len(curr_indxs):
            sampled_all = True
        self.sampled_at_threshs[thresh] = t
        prv_sampled = len(self.all_sampled)
        self.all_sampled = np.union1d(sample, self.all_sampled)
        budget_used = len(self.all_sampled)-prv_sampled
        return sample, budget_used, sampled_all

    def sample_high_low(self, thresh_low, thresh_high, k):
        threshs = (thresh_low, thresh_high)
        if threshs  not in self.sampled_at_threshs:
            self.sampled_at_threshs[threshs] = 0
        
        t = self.sampled_at_threshs[threshs]
        curr_indxs = self.indxs[(self.indxs >=threshs[0])*(self.indxs <threshs[1])]
        #print("sampling fom", curr_indxs, curr_indxs.shape, thresh)
        if t >= len(curr_indxs):
            return curr_indxs,0,True 
        sample = curr_indxs[t:t+k]
        t+=len(sample)
        sampled_all = False
        if t >= len(curr_indxs):
            sampled_all = True
        self.sampled_at_threshs[threshs] = t
        prv_sampled = len(self.all_sampled)
        self.all_sampled = np.union1d(sample, self.all_sampled)
        budget_used = len(self.all_sampled)-prv_sampled
        return sample, budget_used, sampled_all