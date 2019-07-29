from scipy.sparse import issparse, vstack, hstack, csr_matrix as sparse
import numpy as np
from copy import deepcopy

class InvalidRowsAmount(Exception):
    pass


def set_n_buses(ppc, Nhrs):
    try:
        if ppc['bus'].shape[0] % Nhrs != 0:
            raise InvalidRowsAmount
        else:    
            n_buses = int(ppc['bus'].shape[0]/Nhrs)
    except InvalidRowsAmount:
        print('in bus data should be N_hours*N_buses rows')
        raise InvalidRowsAmount
    return n_buses


def get_A_integral(ppc, Nhrs=2):
    """ONLY FOR 2 HRS"""
    gens_hrs = ppc['gen'][:, 0]
    gens_hrs = np.sort(gens_hrs)
    
    n_buses = set_n_buses(ppc, Nhrs)
    n_gens  = len(gens_hrs) // 2
    
    A_v = np.zeros((n_buses, ppc['bus'].shape[0]))
    A_d = np.zeros((n_buses, ppc['bus'].shape[0]))

    A_p = np.zeros((n_buses, ppc['bus'].shape[0]))
    #for i, gen_num in enumerate(gens_hrs[:len(gens_hrs)//2]):
    #    A_p[i,int(gen_num)-1] = 1
    #    A_p[i,int(gen_num)-1 + int(ppc['bus'].shape[0]/Nhrs)] = 1
    for i, row in enumerate(A_p):
        #if (i+1) in gens_hrs:
        row[i] = 1
        row[i + int(ppc['bus'].shape[0]/Nhrs)] = 1
    A_q = np.zeros((n_buses, ppc['bus'].shape[0]))
    res1 = np.hstack((A_v, A_d))
    res2 = np.hstack((A_p, A_q))
    return np.hstack((res1, res2))

def get_l_n_u_inegral(ppc, lower_bound, upper_bound, Nhrs=2):
    """ONLY FOR 2 HRS"""
    """either lower and upper bound must be positive"""
    gens_hrs = ppc['gen'][:, 0]
    gens_hrs = np.sort(gens_hrs)
    
    n_buses = set_n_buses(ppc, Nhrs)
    n_gens  = len(gens_hrs) // 2    
    l = np.zeros(n_buses)
    u = np.zeros(n_buses)
    for i in range(len(l)):
        if (i+1) in gens_hrs:
            l[i] = lower_bound
            u[i] = upper_bound
        else:
            l[i] = -np.inf
            u[i] =  np.inf
    return l, u

def get_A_ramping(ppc, Nhrs=2):
    """ONLY FOR 2 HRS"""
    gens_hrs = ppc['gen'][:, 0]
    gens_hrs = np.sort(gens_hrs)
    
    n_buses = set_n_buses(ppc, Nhrs)
    n_gens  = len(gens_hrs) // 2
    A_v = np.zeros((n_buses, ppc['bus'].shape[0]))
    A_d = np.zeros((n_buses, ppc['bus'].shape[0]))

    A_p = np.zeros((n_buses, ppc['bus'].shape[0]))
    #for i, gen_num in enumerate(gens_hrs[:len(gens_hrs)//2]):
    #    A_p[i,int(gen_num)-1] = -1
    #    A_p[i,int(gen_num)-1 + int(ppc['bus'].shape[0]/Nhrs)] = 1
    for i, row in enumerate(A_p):
        #if (i+1) in gens_hrs:
        row[i] = -1
        row[i + int(ppc['bus'].shape[0]/Nhrs)] = 1
    A_q = np.zeros((n_buses, ppc['bus'].shape[0]))
    res1 = np.hstack((A_v, A_d))
    res2 = np.hstack((A_p, A_q))
    return np.hstack((res1, res2))


def get_l_n_u_ramping(ppc, lower_bound, upper_bound, Nhrs=2):
    """ONLY FOR 2 HRS"""
    """lower_bound must be neagtiva, upper_bound must be positive"""
    gens_hrs = ppc['gen'][:, 0]
    gens_hrs = np.sort(gens_hrs)
    
    n_buses = set_n_buses(ppc, Nhrs)
    n_gens  = len(gens_hrs) // 2
    l = np.zeros(n_buses)
    u = np.zeros(n_buses)
    for i in range(len(l)):
        if (i+1) in gens_hrs:
            l[i] = lower_bound
            u[i] = upper_bound
        else:
            l[i] = -np.inf
            u[i] =  np.inf
    return l, u

def generate_Nhrs_cases(ppc, Nhrs=2, mul_dem=1.5, mul_gen=1.5):
    """
    muls are the factors for increasing or decreasing demand and generation
    """
    ppc_Nhrs = deepcopy(ppc)
    to_copy_keys = list(ppc_Nhrs.keys())
    to_copy_keys.remove('version')
    to_copy_keys.remove('baseMVA')
    for i in range(Nhrs-1):
        for name in to_copy_keys:
            tmp = deepcopy(ppc_Nhrs[name])
            if name == 'bus' or name == 'gen':
                if name == 'bus':
                    tmp[:,[2,3]] *= mul_dem
                if name == 'gen':
                    tmp[:,1] *= mul_gen
                tmp[:,0] += 30
            if name == 'branch':
                tmp[:,[0,1]] += 30
            ppc_Nhrs[name] = np.vstack((ppc_Nhrs[name], tmp))
    return ppc_Nhrs
    

def set_hours_constr(ppc, bounds_int, bounds_ram, Nhrs=2):
    A_int = get_A_integral(ppc, Nhrs=Nhrs)
    A_ram = get_A_ramping(ppc, Nhrs=Nhrs)
    
    l_int, u_int = get_l_n_u_inegral(ppc, *bounds_int, Nhrs=Nhrs)
    l_ram, u_ram = get_l_n_u_ramping(ppc, *bounds_ram, Nhrs=Nhrs)
    
    ppc['A'] = sparse(np.vstack((A_int, A_ram)))
    ppc['u'] = np.hstack((u_int, u_ram))
    ppc['l'] = np.hstack((l_int, l_ram))

    return ppc
