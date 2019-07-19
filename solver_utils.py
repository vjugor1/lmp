import numpy as np
from numpy.linalg import matrix_rank
from solution_options import *
from scipy.optimize import minimize
import cvxpy as cp

get_idx_gencost = lambda results, threshold: (results['gen'][:, 0][results['gencost'][:, 4] > threshold] - 1).astype(int)


def Rank_info(matrix):
    print('#Var = %d'% matrix.shape[1])
    print('#Eq = %d'% matrix.shape[0])
    r = matrix_rank(matrix)
    #print('Rank A = %d'% r)
    print('Underdetermined: ', matrix.shape[1] >= r)
    

def mu_sigma_pi(muls):
    mu = np.concatenate((muls['mu_v_max'], muls['mu_v_min']))
    sigma = np.concatenate((muls['mu_s_from'], muls['mu_s_to']))
    pi = np.concatenate((muls['mu_p_max'], muls['mu_p_min']))
    #rho = np.concatenate((muls['mu_q_max'], muls['mu_q_min']))
    return mu, sigma, muls['mu_p_max'], muls['mu_p_min'], muls['mu_q_max'], muls['mu_q_min']


def A_sigmas(Js, sigma, mode, A_blocks, b, xs):
    if mode == fixed:
        b -= np.dot(Js.T.toarray(), sigma)
    elif mode == zero_fixed:
        idx_sigma = sigma > 0
        A_blocks.append((Js.T[:, idx_sigma]).toarray())
        xs['sigma'] = A_blocks[-1].shape[1]
    else:
        A_blocks.append((Js.T).toarray())
        xs['sigma'] = A_blocks[-1].shape[1]

        
def A_mu(mu, mode, A_blocks, b, xs):
    
    Nbus = len(mu)//2
    if mode == fixed:
        b -= mu
    elif mode == zero_fixed:
        idx_mu = mu > 0
        Jmu = np.block([[np.zeros((Nbus, Nbus)), np.zeros((Nbus, Nbus))], 
                        [np.eye(Nbus), -np.eye(Nbus)]])
        A_blocks.append(Jmu[:, idx_mu])
        xs['mu'] = A_blocks[-1].shape[1]
    else:
        Jmu = np.block([[np.zeros((Nbus, Nbus)), np.zeros((Nbus, Nbus))], 
                        [np.eye(Nbus), -np.eye(Nbus)]])
        A_blocks.append(Jmu)
        xs['mu'] = A_blocks[-1].shape[1]
        
        
def ModCost(C, idx, new_C):
    tmp_C = C.copy()
    tmp_C[idx] = new_C
    return tmp_C


def LS_NNLS(A, b, alg, verbose=False):
    x = cp.Variable(A.shape[1])
    cost = cp.sum_squares(A*x - b)
    if alg == LS:
        prob = cp.Problem(cp.Minimize(cost))
        #res = np.linalg.lstsq(A, b)
    else:
        prob = cp.Problem(cp.Minimize(cost), [x >= np.zeros_like(x)])
        #res = nnls(A, b)
    prob.solve()    
    if verbose:
        print('f(x^*) = %f' % prob.value)
    return np.array(x.value).ravel()#res[0]

def B_LS(A, b, bounds, threshold, verbose=False):
    #x = cp.Variable(A.shape[1])
    #cost = cp.sum_squares(A*x - b)
    #b_min, b_max = bounds
    #constraints = []
    #if any(b_min != None):
    #    constraints.append(x[b_min != None] >= b_min[b_min != None])
    #if any(b_max != None):
    #    constraints.append(x[b_max != None] <= b_max[b_max != None])
    
    #prob = cp.Problem(cp.Minimize(cost), constraints)
    #prob.solve(solver=cp.ECOS_BB, verbose=True)
    f = lambda x: np.dot(np.dot(A, x) - b, np.dot(A, x) - b)
    np.random.seed(0)
    x0 = threshold*np.random.rand(A.shape[1])#np.random.randint(low=0., high=threshold, size=A.shape[1])
    res = minimize(f,
                   x0,
                   method='SLSQP',
                   bounds=bounds,
                   tol=1e-8,
                   options={'maxiter': 10**10,})
    if verbose:
        print('f(x^*) =', res.fun)
    #print(np.array(x.value).ravel())    
    return res.x#np.array(x.value).ravel()#