import numpy as np

def __sigma_squared(i, obs):
    mu_hat = (1/2+obs[:i].sum())/(i+1)
    return (1/4+((obs[:i]-mu_hat)**2).sum())/(i+1) 

def __get_lambda(alpha, obs, i, fixed_sample_size):
    if fixed_sample_size:
        return np.sqrt(2*np.log(2/alpha)/(len(obs)*__sigma_squared(i-1, obs)))
    else:
        return np.sqrt(2*np.log(2/alpha)/(i*np.log(i+1)*__sigma_squared(i-1, obs)))

def __k_plus(obs, target, alpha, trunc_scale, theta, without_replacement=False, N=0, fixed_sample_size=True):
    if without_replacement:
        assert N>0
        t = np.arange(1, len(obs) + 1)
        S_t = np.cumsum(obs)
        S_tminus1 = np.append(0, S_t[0 : (len(obs) - 1)])
        m_wor_i = (N * target - S_tminus1) / (N - (t - 1))
    else:
        m_wor_i = np.repeat(target,len(obs))

    k_t = 1
    for i in range(1, len(obs)+1):
        l = __get_lambda(alpha, obs, i, fixed_sample_size)
        l = np.minimum(l, trunc_scale / m_wor_i[i-1])

        k_t *= (1+l*(obs[i-1]-m_wor_i[i-1]))
        if theta*k_t >= 1/alpha:
            return False
    return True

def __k_minus(obs, target, alpha, trunc_scale, theta, without_replacement=False, N=0, fixed_sample_size=True):
    if without_replacement:
        assert N>0
        m_wor_i = np.concat([np.array([target]), (N*target-np.cumsum(obs[:-1]))/(N-(np.arange(len(obs)-1)))])
    else:
        m_wor_i = np.repeat(target,len(obs))

    k_t = 1
    for i in range(1, len(obs)+1):
        l = __get_lambda(alpha, obs, i, fixed_sample_size)
        #print('l',l)
        l = np.minimum(l, trunc_scale / (1 - m_wor_i[-1]))

        k_t *= (1-l*(obs[i-1]-m_wor_i[i-1]))
        if (1-theta)*k_t >= 1/alpha:
            return False
    return True
        
def __mean_is_in_conf(obs, target, alpha, theta, trunc_scale, fixed_sample_size=True, without_replacement=False, N=0):
    kt_plus = __k_plus(obs, target, alpha, trunc_scale, theta, without_replacement, N, fixed_sample_size)
    if not kt_plus or theta==1:
        return kt_plus
    kt_minus = __k_minus(obs, target, alpha, trunc_scale, theta, without_replacement, N, fixed_sample_size)
    return kt_minus

def test_if_true_mean_is_above_m(obs, m, alpha, fixed_sample_size=True, without_replacement=False, N=0):
    assert not without_replacement or (without_replacement and N>0)
    return __mean_is_in_conf(obs, m, alpha, 1, 3/4, fixed_sample_size, without_replacement, N)

def test_if_true_mean_is_below_m(obs, m, alpha, fixed_sample_size=True, without_replacement=False, N=0):
    assert not without_replacement or (without_replacement and N>0)
    return __mean_is_in_conf(obs, m, alpha, 0, 3/4, fixed_sample_size, without_replacement, N)
    