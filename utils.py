from pandapower.plotting.collections import create_bus_collection
from pandapower.plotting.collections import create_line_collection
from pandapower.plotting.collections import draw_collections
from pandapower.plotting.collections import create_annotation_collection
from pandapower.plotting.collections import create_gen_collection
from pandapower.plotting.collections import create_ext_grid_collection
from matplotlib.font_manager import FontProperties

from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PQ, REF
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS
from pypower.idx_bus import BUS_TYPE, REF, VM, VA, MU_VMAX, MU_VMIN, LAM_P, LAM_Q
from pypower.idx_brch import F_BUS, T_BUS, RATE_A, PF, QF, PT, QT, MU_SF, MU_ST
from pypower.idx_gen import GEN_BUS, PG, QG, VG, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN
from pypower.idx_cost import MODEL, PW_LINEAR, NCOST
import numpy as np
import matplotlib.pyplot as plt

from numpy import r_, c_, ix_, zeros, pi, ones, exp, argmax

from pypower.api import opf_consfcn, opf_costfcn

from pypower.api import opf_setup

from pypower.api import ext2int
from pypower.api import makeYbus



def get_Jac(ppc, ppopt, results):
    ppc = ext2int(ppc)
    baseMVA, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]

    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    #print(Ybus.shape)
    #print(Yf.shape)
    #print(Yt.shape)

    x = results['x']
    om = opf_setup(ppc, ppopt)

    g, geq, dg, dgeq = opf_consfcn(x, om, Ybus, Yf, Yt, ppopt)

    return dg.T, dgeq.T


def get_muls(results, Nhrs=1):
    res_struct = {}

    bus    = results['bus']
    branch = results['branch']

    res_struct['mu_v_max'] = results['bus'][:, MU_VMAX]
    res_struct['mu_v_min'] = results['bus'][:, MU_VMIN]

    res_struct['mu_s_from']   = results['branch'][:, MU_SF]
    res_struct['mu_s_to']   = results['branch'][:, MU_ST]
    
    res_struct['lam_p'] = results['bus'][:, LAM_P]
    res_struct['lam_q'] = results['bus'][:, LAM_Q]
    
    results['gen_sorted'] = np.concatenate([results['gen'], results['gencost'][:, 4][:, None]], axis=1)
    results['gen_sorted'] = results['gen_sorted'][results['gen_sorted'][:, 0].argsort()]
    
    if Nhrs > 1:
        n_buses = results['lin']['mu']['l']['usr'].shape[0]
        res_struct['ls_int'] = results['lin']['mu']['l']['usr'][:int(n_buses/Nhrs)]
        res_struct['ls_ram'] = results['lin']['mu']['l']['usr'][int(n_buses/Nhrs):]
        
        res_struct['us_int'] = results['lin']['mu']['u']['usr'][:int(n_buses/Nhrs)]
        res_struct['us_ram'] = results['lin']['mu']['u']['usr'][int(n_buses/Nhrs):]
        
        res_struct['Nhrs'] = Nhrs
    
    res_struct['mu_p_max'] = results['gen_sorted'][:, MU_PMAX]
    res_struct['mu_p_min'] = results['gen_sorted'][:, MU_PMIN]

    res_struct['mu_q_max'] = results['gen_sorted'][:, MU_QMAX]
    res_struct['mu_q_min'] = results['gen_sorted'][:, MU_QMIN]
    
    res_struct['C'] = results['gen_sorted'][:, -1]
    return res_struct


def is_price_forming(gen, gen_cost, verbose=False, Nhrs=1, results=None):
    #return (bus numbers - 1)
    assert Nhrs > 0, "Nhrs should be positive"
    
    if Nhrs == 1:
        idx = []
        for i in range(len(gen)):
            if gen_cost[i, 4] > 0 and gen[:, MU_PMAX][i] == 0 and gen[:, MU_PMIN][i] == 0:
                idx.append(i)
                if verbose:
                    print('Price forming bus #%d\t C_g=%f' % (gen[i, 0], gen_cost[i, 4]))
    
    else:
        if results == None:
            raise Exception('results dict is needed to analyze several hours case')
        idx = []
        muls_dict = get_muls(results, Nhrs)
        num_of_buses = int(len(gen) / Nhrs)
        for hour in range(Nhrs):
            for i in range(num_of_buses):
                if hour == 0:
                    if gen_cost[i, 4] > 0 and gen[:, MU_PMAX][i] == 0 and gen[:, MU_PMIN][i] == 0:
                        idx.append(i)
                        if verbose:
                            print('Hour %d:Price forming bus #%d\t C_g=%f' % (hour+1, gen[i, 0], gen_cost[i, 4]))
                else:
                    
                    
                    logical_muls = gen[:, MU_PMAX][i + num_of_buses*hour] == 0 and gen[:, MU_PMIN][i+ num_of_buses*hour] == 0 and muls_dict['ls_int'][i + num_of_buses*(hour-1)] == 0 and muls_dict['ls_ram'][i + num_of_buses*(hour-1)] == 0 and muls_dict['us_int'][i + num_of_buses*(hour-1)] == 0 and muls_dict['us_ram'][i + num_of_buses*(hour-1)] == 0
                    
                    
                    if gen_cost[i + num_of_buses * hour, 4] > 0. and logical_muls == True:
                        idx.append(i + num_of_buses * hour)
                        if verbose:
                            print('Hour %d:Price forming bus #%d\t C_g=%f' % (hour+1, gen[i, 0], gen_cost[i + num_of_buses * hour, 4]))
                        
                        
                
        #for i in range(len(gen)):
            #logical_muls = gen[:, MU_PMAX][i] == 0 and gen[:, MU_PMIN][i] == 0 and muls_dict['ls_int'][i] == 0 and muls_dict['ls_ram'][i] == 0 and muls_dict['us_int'][i] == 0 and muls_dict['us_ram'][i] == 0
            #if gen_cost[i, 4] > 0 and logical_muls:
            #    idx.append(i)
            #    if verbose:
            #        print('Price forming bus #%d\t C_g=%f' % (gen[i, 0], gen_cost[i, 4]))
    
    return [int(gen[i, 0] - 1) for i in idx]


Median_func = lambda **kwargs: np.median(kwargs['gen_cost'][:, 4][(kwargs['gen_cost'][:, 4] > 0)* (kwargs['gen_cost'][:, 4] < kwargs['threshold'])])

Rand_replace = lambda **kwargs: np.random.randint(low=kwargs['alpha']*kwargs['threshold'], high=kwargs['beta']*kwargs['threshold'])


def ReplaceLambda(bus, gen_cost, forming_idx, non_cr_only, threshold, replace_func, **kwargs):
    tmp = bus[:, LAM_P].copy()
    idx = []
    kwargs['gen_cost'] = gen_cost
    kwargs['threshold'] = threshold

    for i in forming_idx:
        if bus[i, LAM_P] > threshold:
            #tmp[i] = alpha*np.random.randint(low=0., high=threshold)#alpha*threshold
            tmp[i] = replace_func(**kwargs)
        else:
            idx.append(i)
    return (tmp, forming_idx) if not non_cr_only else (tmp, idx)

def gen_C_vecs(Nbus, gen, gen_cost):
    res = np.zeros(Nbus)
    res[(gen[:, 0] - 1).astype(int)] = gen_cost[:, 4]
    C_g, C_d = np.zeros(Nbus), np.zeros(Nbus)
    C_g[res > 0] = res[res > 0]
    C_d[res < 0] = res[res < 0]
    return C_g, C_d

def recompute_costs(results):
    bus = results['bus']
    gen = results['gen']
    
    gen_buses = gen[:,0]
    lambdas = []
    mu_pmax = gen[:, MU_PMAX]
    mu_pmin = gen[:, MU_PMIN]

    for bus_num in gen_buses:
        for i, row in enumerate(bus):
            if bus_num == row[0]:
                lambdas.append(bus[i, -1])
                #lambdas.append(bus[i,LAM_P])
                
    costs = lambdas*np.sign(gen_C_vec(gen, results['gencost'])) - mu_pmax + mu_pmin
    return costs


import pypower.idx_gen as idx_gen
import pypower.idx_bus as idx_bus
import pypower.idx_brch as idx_brch
import pypower.idx_cost as idx_cost

blacklist = ['PQ', 'PV', 'REF', 'NONE', 'PW_LINEAR', 'POLYNOMIAL']


def get_module_locals(module, blacklist):
    tmp = [(key, value) for key, value in vars(module).items() if not key.startswith("__") and key not in blacklist]
    tmp.sort(key=lambda x: x[-1])
    return [i[0] for i in tmp]

module_list = [idx_bus, idx_gen, idx_brch, idx_cost]


def get_cols(module_list=module_list, blacklist=blacklist):
    return (get_module_locals(module, blacklist) for module in module_list)

