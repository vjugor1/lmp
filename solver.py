import numpy as np
from collections import OrderedDict

from solution_options import *
from solver_utils import *
import utils
from utils import Rand_replace, Median_func

def_opts = {'mode': separate_mode, 
            'lam_q': lam_q_fixed, 
            'mu': zero_fixed, 
            'sigma': zero_fixed,
            'pi_psi': fixed,
            'rho': fixed}


def Ab(results, J, Js, lam_q, idx_to_replace, new_lambdas, sigma, mu, pi_max, pi_min, rho, Cg, Cd, C_full_sorted, opt):
    
    xs = OrderedDict()
    
    Nbus = len(lam_q)
    JpT, JqT = J[:Nbus, :].T, J[Nbus:, :].T
    JpT, JqT = JpT.toarray(), JqT.toarray()
    
    #b = -np.dot(JpT[:, idx_to_replace], new_lambdas)
    b = 0.
    A_blocks = []
    idx_lam_p = np.setdiff1d(np.arange(Nbus), idx_to_replace)
    
    if opt['lam_q'] == lam_q_fixed:
        b -= np.dot(JqT, lam_q)
    
    if opt['mode'] == separate_mode:
        xs['lam_p'] = Nbus - len(idx_to_replace)
        
        b -= np.dot(JpT[:, idx_to_replace], new_lambdas)
        if opt['lam_q'] == lam_q_fixed:
            A_blocks.append(JpT[:, idx_lam_p])
        else:
            #print('here')
            A_blocks.append(np.concatenate([JpT[:, idx_lam_p], JqT], axis=1))
            xs['lam_q'] = len(lam_q)
    
    elif opt['mode'] == simultaneous_mode:
        
        Cg_mod = ModCost(Cg, idx_to_replace, new_lambdas)
        
        b -= np.dot(JpT, Cg_mod - Cd)/2
        
        Cg_mod[idx_to_replace] = -1
        
        if opt['pi_psi'] == all_new:
            #print(Cg_mod)
            #print(np.nonzero(Cg_mod > 0)[0])
            #print(np.nonzero(Cd < 0)[0])
            A_blocks.append(np.concatenate((JpT[:, np.nonzero(Cg_mod > 0)[0]], 
                                            JpT[:, np.nonzero(Cd < 0)[0]], 
                                            -JpT[:, np.nonzero(Cg_mod > 0)[0]], 
                                            -JpT[:, np.nonzero(Cd < 0)[0]]), axis=1)/2)
            
            idx_remain = np.setdiff1d(np.arange(Nbus), 
                                      np.union1d(np.nonzero(Cg > 0)[0], np.nonzero(Cd < 0)[0]))
            
            A_blocks.append(JpT[:, idx_remain])
            
            xs['lam_p'] = len(idx_remain)
            xs['pi_max'] = xs['pi_min'] = sum(Cg_mod > 0)
            xs['psi_max'] = xs['psi_min'] = sum(Cd < 0)
        
        else:
            idx_max = np.nonzero(pi_max > 0)[0]
            idx_min = np.nonzero(pi_min > 0)[0]
            
            idx_pi_max = np.intersect1d(idx_max, np.nonzero(C_full_sorted > 0)[0])
            idx_pi_min = np.intersect1d(idx_min, np.nonzero(C_full_sorted > 0)[0])
            
            idx_psi_max = np.intersect1d(idx_max, np.nonzero(C_full_sorted < 0)[0])
            idx_psi_min = np.intersect1d(idx_min, np.nonzero(C_full_sorted < 0)[0])
            
            
            A_blocks.append(np.concatenate((JpT[:, results['gen_sorted'][idx_pi_max, 0].astype(int) - 1], 
                                            JpT[:, results['gen_sorted'][idx_psi_max, 0].astype(int) - 1], 
                                            -JpT[:, results['gen_sorted'][idx_pi_min, 0].astype(int) - 1], 
                                            -JpT[:, results['gen_sorted'][idx_psi_min, 0].astype(int) - 1]), axis=1)/2)
            
            idx_remain = np.setdiff1d(np.arange(Nbus), 
                                      np.union1d(results['gen_sorted'][np.nonzero(C_full_sorted > 0)[0], 0].astype(int) - 1, 
                                                 results['gen_sorted'][np.nonzero(C_full_sorted < 0)[0], 0].astype(int) - 1))
            
            A_blocks.append(JpT[:, idx_remain])
            
            xs['lam_p'] = len(idx_remain)
            xs['pi_max'] = len(idx_pi_max)
            xs['psi_max'] = len(idx_psi_max)
            xs['pi_min'] = len(idx_pi_min)
            xs['psi_min'] = len(idx_psi_min)
        
        if opt['lam_q'] != lam_q_fixed:
            #if opt['rho'] == zero_fixed:
            #    print('TODO')
            #elif opt['rho'] == all_new:
                #idx_rho = np.arange(2*Nbus)
            #else:
            #    idx_rho = np.arange(Nbus)
            idx_rho = np.arange(Nbus)    
            A_blocks.append(np.concatenate([JqT, -JqT], axis=1)[:, idx_rho])
            if opt['rho'] == fixed:
                xs['lam_q'] = JqT.shape[-1]
            else:
                xs['rho'] = 2*JqT.shape[-1]
    else:
        print('MODE ERROR')
        A, b = None, None
    
    A_sigmas(Js, sigma, opt['sigma'], A_blocks, b, xs)
    A_mu(mu, opt['mu'], A_blocks, b, xs)
    
    A = np.concatenate(A_blocks, axis=1)
    
    return A, b, xs


solution_options = {'mode': separate_mode, 
                    'lam_q': lam_q_fixed, 
                    'mu': all_new, 
                    'sigma': all_new,
                    'pi_psi': all_new,
                    'rho': all_new}

def Solve(dgT, dgeqT, results, threshold, 
          replace_func=Median_func,
          #case=0,
          opt=def_opts,
          verbose=True,
          alg=NNLS, 
          idx_mode=all_forming, **kwargs):
    
    Nbus      = results['bus'].shape[0]
    muls      = utils.get_muls(results)
    bus_idx   = utils.is_price_forming(results['gen'], results['gencost'])
    
    Js, J     = dgT[:, :2*Nbus], dgeqT[:, :2*Nbus]
    
    Cg, Cd = utils.gen_C_vecs(Nbus, results['gen'], results['gencost'])
    
    mu, sigma, pi_max, pi_min, rho_max, rho_min = mu_sigma_pi(muls)
    new_lambdas, idx_to_replace = utils.ReplaceLambda(results['bus'],
                                                      results['gencost'],
                                                      bus_idx, idx_mode,
                                                      threshold,
                                                      replace_func, **kwargs)
    lam_p_idx = Nbus - len(idx_to_replace)

    rho_gen = np.concatenate((rho_max[results['gencost'][:, 4] > 0], rho_min[results['gencost'][:, 4] > 0]))
    #idx_criminals = list(set(idx_to_replace) & set(get_idx_gencost(results, threshold)))
    idx_gen = get_idx_gencost(results, 0)
    
    A, b, xs = Ab(results, J, Js, muls['lam_q'], 
                  idx_to_replace, 
                  #idx_criminals,
                  new_lambdas[idx_to_replace], 
                  sigma, mu, pi_max, pi_min, rho_gen, 
                  Cg, Cd, muls['C'],
                  opt)
    #print(muls['C'])
    
    if verbose:
        #print(idx_criminals)
        Rank_info(A)
        print(xs)
    
    if alg in [LS, NNLS]:
        x = LS_NNLS(A, b, alg, verbose=verbose)
    
    elif alg == 2:
        if opt['mode'] == separate_mode:
            if opt['lam_q']:
                idx_lam_pq = lam_p_idx + Nbus
            else:
                idx_lam_pq = lam_p_idx

            b_min = np.zeros(A.shape[1])
            b_min[: idx_lam_pq] = [None]*idx_lam_pq

            b_max = [None]*A.shape[1]

            x = B_LS(A, b, tuple(zip(b_min, b_max)), threshold, verbose=verbose)
        elif opt['mode'] != separate_mode:
            idx_pi_psi = xs['pi_max'] + xs['psi_max'] + xs['pi_min'] + xs['psi_min']
            b_min = np.zeros(A.shape[1])
            b_min[idx_pi_psi: idx_pi_psi + xs['lam_p']] = [None]*xs['lam_p']
            
            if opt['lam_q'] == lam_q_recompute:
                b_min[idx_pi_psi + xs['lam_p']: idx_pi_psi + xs['lam_p'] + Nbus] = [None]*Nbus

            b_max = [None]*A.shape[1]

            x = B_LS(A, b, tuple(zip(b_min, b_max)), threshold, verbose=verbose)
        else:
            print('Setting Error')
            x = np.zeros(A.shape[1])
    
    elif alg == 3 and opt['mode'] == separate_mode:
        C = Cg + Cd
        c = C[np.setdiff1d(np.arange(Nbus), bus_idx)]
        
        b_min = np.zeros(A.shape[1])
        
        #print(c > 0)
        b_min[:len(c)][c > 0] = c[c > 0]
        
        b_max = np.array([None]*A.shape[1])
        #print(len(c))
        b_max[:len(c)][c < 0] = abs(c[c < 0])
        b_max[lam_p_idx: ] = [None]*(len(b_max) - lam_p_idx)
        
        
        x = B_LS(A, b, tuple(zip(b_min, b_max)), threshold, verbose)
    else:
        print('Setting Error')
        x = np.zeros(A.shape[1])
        
    res_struct = {}
    if opt['mode'] == separate_mode:
        res_lambdas = np.zeros(Nbus)
        res_lambdas[idx_to_replace] = new_lambdas[idx_to_replace]
        res_lambdas[np.setdiff1d(range(Nbus), idx_to_replace)] = x[: lam_p_idx]
        res_struct['lam_p'] = res_lambdas
        idx = lam_p_idx
        if opt['lam_q'] != lam_q_fixed:
            idx += xs['lam_q']
            res_struct['lam_q'] = x[lam_p_idx: idx]
            
        if opt['sigma'] == zero_fixed:
            new_sigma = np.zeros_like(sigma)
            new_sigma[sigma > 0] = x[idx: idx + xs['sigma']]
            idx += xs['sigma']
            res_struct['sigma'] = new_sigma
            
        elif opt['sigma'] == all_new:
            res_struct['sigma'] = x[idx: idx + xs['sigma']]
            idx += xs['sigma']
            
        if opt['mu'] == zero_fixed:
            new_mu = np.zeros_like(mu)
            new_mu[mu > 0] = x[idx:]
            res_struct['mu'] = new_mu
        
        elif opt['mu'] == all_new:
            res_struct['mu'] = x[idx:]
                    
        '''
        Cg_mod = ModCost(Cg, idx_to_replace, new_lambdas[idx_to_replace])
        #Cg_mod = Cg_mod[np.setdiff1d(np.arange(Nbus), idx_to_replace)]
        c = Cg_mod + Cd - 2*res_lambdas#[np.setdiff1d(np.arange(Nbus), idx_to_replace)]
        #C = np.concatenate([Cg, Cd])
        #l = np.concatenate([-res_lambdas, -res_lambdas])
        Cg_mod = Cg_mod[np.setdiff1d(np.arange(Nbus), idx_to_replace)]
        I = np.eye(Nbus)
        B = np.block([I[:,  np.nonzero(Cg_mod > 0)[0]], 
                      I[:,  np.nonzero(Cd < 0)[0]], 
                      -I[:,  np.nonzero(Cg_mod > 0)[0]], 
                      -I[:,  np.nonzero(Cd < 0)[0]]])
        
        if opt['rho'] == zero_fixed:
            idx_rho = rho > 0
        elif opt['rho'] == all_new:
            idx_rho = np.arange(Nbus)
        #B = np.concatenate((-np.eye(Nbus), np.eye(Nbus)), axis=1)
        #B = np.block([[B, np.zeros_like(B)], 
        #              [np.zeros_like(B), B]])
        #c = C + l
        x_pi = LS_NNLS(B, c, verbose)
        
        res_struct['pi_psi_max'] = x_pi[:sum(Cg_mod > 0) + sum(Cd < 0)]
        res_struct['pi_psi_min'] = x_pi[sum(Cg_mod > 0) + sum(Cd < 0):]
        #res_struct['pi_max'] = x_pi[:Nbus]
        #res_struct['pi_min'] = x_pi[Nbus:2*Nbus]
        #res_struct['psi_max'] = x_pi[2*Nbus:3*Nbus]
        #res_struct['psi_min'] = x_pi[3*Nbus:]
        
        if verbose:
            print(len(x_pi))
            print(res_struct['pi_psi_max']*res_struct['pi_psi_min'])
            #print(res_struct['pi_max']*res_struct['pi_min'])
            #print(res_struct['psi_max']*res_struct['psi_min'])
        '''
    else:
        Cg_mod = ModCost(Cg, idx_to_replace, new_lambdas[idx_to_replace])
        res_lambdas = np.zeros(Nbus)
        res_lambdas[idx_to_replace] = new_lambdas[idx_to_replace]
        if opt['pi_psi'] == all_new:
            res_lambdas[np.setdiff1d(idx_gen, idx_to_replace)] = Cg_mod[np.setdiff1d(idx_gen, idx_to_replace)] + x[:xs['pi_max']] 
            res_lambdas[np.setdiff1d(idx_gen, idx_to_replace)] -= x[xs['pi_max'] + xs['psi_max']: xs['psi_max'] + xs['pi_max'] + xs['pi_min']]
            res_lambdas[Cd < 0] = -Cd[Cd < 0] + x[xs['pi_max']: xs['pi_max'] + xs['psi_max']] 
            res_lambdas[Cd < 0] -= x[xs['pi_max'] + xs['psi_max'] + xs['pi_min']: xs['pi_max'] + xs['psi_max'] + xs['pi_min'] + xs['psi_min']]
            
            idx_remain = np.setdiff1d(np.arange(Nbus), 
                                      np.union1d(np.nonzero(Cg_mod > 0)[0], np.nonzero(Cd < 0)[0]))
            
            #print(len(idx_remain))
            
            res_lambdas[idx_remain] = x[xs['pi_max'] + xs['psi_max'] + xs['pi_min'] + xs['psi_min']: xs['pi_max'] + xs['psi_max'] + xs['pi_min'] + xs['psi_min'] + xs['lam_p']]
        else:
            idx_max = np.nonzero(pi_max > 0)[0]
            idx_min = np.nonzero(pi_min > 0)[0]
            
            idx_pi_max = np.intersect1d(idx_max, np.nonzero(muls['C'] > 0)[0])
            idx_pi_min = np.intersect1d(idx_min, np.nonzero(muls['C'] > 0)[0])
            
            idx_psi_max = np.intersect1d(idx_max, np.nonzero(muls['C'] < 0)[0])
            idx_psi_min = np.intersect1d(idx_min, np.nonzero(muls['C'] < 0)[0])
            
            idx_remain = np.setdiff1d(np.arange(Nbus), 
                                      np.union1d(results['gen_sorted'][np.nonzero(muls['C'] > 0)[0], 0].astype(int) - 1,
                                                 results['gen_sorted'][np.nonzero(muls['C'] < 0)[0], 0].astype(int) - 1))
            
            #print(idx_gen)
            #print(np.nonzero(muls['C'] > 0)[0])
            #print(np.intersect1d(idx_remain, idx_gen))
            #print(idx_pi_max)
            #print(idx_pi_min)
            
            res_lambdas[results['gen_sorted'][idx_pi_max, 0].astype(int) - 1] += x[:xs['pi_max']]
            #print(idx_psi_max)
            #print(x[xs['pi_max']:xs['pi_max'] + xs['psi_max']])
            #print(res_lambdas[idx_psi_max])
            res_lambdas[results['gen_sorted'][idx_psi_max, 0].astype(int) - 1] += x[xs['pi_max']:xs['pi_max'] + xs['psi_max']]
            #print(res_lambdas[idx_psi_max])
            res_lambdas[results['gen_sorted'][idx_pi_min, 0].astype(int) - 1] -= x[xs['pi_max'] + xs['psi_max']:xs['pi_max'] + xs['psi_max'] + xs['pi_min']]
            #print(idx_psi_min)
            #print(x[xs['pi_max'] + xs['psi_max'] + xs['pi_min']: xs['pi_max'] + xs['psi_max'] + xs['pi_min'] + xs['psi_min']])
            res_lambdas[results['gen_sorted'][idx_psi_min, 0].astype(int) - 1] -= x[xs['pi_max'] + xs['psi_max'] + xs['pi_min']: xs['pi_max'] + xs['psi_max'] + xs['pi_min'] + xs['psi_min']]
            
            res_lambdas[np.setdiff1d(idx_gen, idx_to_replace)] += Cg_mod[np.setdiff1d(idx_gen, idx_to_replace)]
            res_lambdas[np.setdiff1d(np.arange(Nbus), idx_gen)] -= Cd[np.setdiff1d(np.arange(Nbus), idx_gen)]
            
            res_lambdas[idx_remain] += x[xs['pi_max'] + xs['psi_max'] + xs['pi_min'] + xs['psi_min']: xs['pi_max'] + xs['psi_max'] + xs['pi_min'] + xs['psi_min'] + xs['lam_p']]
            #print(res_lambdas[idx_remain])
            
        res_struct['lam_p'] = res_lambdas
        
        if verbose:
            pass
            #print('MU_PMAX')
            #print(len(x[:xs['pi_max_min'] + xs['psi_max_min']]))
            #print(x[:xs['pi_max'] + xs['psi_max']])
            #print(results['gen'][:, MU_PMAX])
            
            #print('MU_PMIN')
            #print(x[xs['pi_max'] + xs['psi_max']: 2*(xs['pi_max'] + xs['psi_max'])])
            #print(results['gen'][:, MU_PMAX + 1])
            
            #print(x[:xs['pi_max_min'] + xs['psi_max_min']]*x[xs['pi_max_min'] + xs['psi_max_min']: 2*(xs['pi_max_min'] + xs['psi_max_min'])])
    results['new_lam'] = res_struct['lam_p']
    return res_struct


def Solver(ppc, ppopt, results, threshold=2000, verbose=True):
    dgT, dgeqT = utils.get_Jac(ppc, ppopt, results)
    if verbose:
        utils.is_price_forming(results['gen'], 
                               results['gencost'], 
                               verbose=verbose)
    sol_opt = set_options()
    alg = set_alg()
    idx_mode = set_idx_mode()
    alg = check_opt_alg(sol_opt, alg)
    
    res_struct = Solve(dgT, dgeqT, results, threshold, 
                   replace_func=Median_func, 
                   opt=sol_opt,
                   alg=alg, 
                   idx_mode=idx_mode,
                   alpha=0.2, beta=0.8)
    
    return res_struct

