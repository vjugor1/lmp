from collections import OrderedDict
from IPython.display import clear_output

import itertools

separate_mode = 0
simultaneous_mode = 1

lam_q_fixed = 0
lam_q_recompute = 1

fixed = 0
zero_fixed = 1
all_new = 2

mode_list = [separate_mode, simultaneous_mode]
lam_q_list = [lam_q_fixed, lam_q_recompute]
duals_list = [fixed, zero_fixed, all_new]
pi_psi_list = [zero_fixed, all_new]
#my_var_name = [ k for k,v in locals().iteritems() if v == my_var][0]

LS = 0
NNLS = 1
PQ_R = 2
BoundedLS = 3
linprog = 4

class InvalidSettings(Exception):
    pass

def key_val_is_in(d, key, val_list):
    try:
        if d[key] not in val_list:
            raise InvalidSettings
            
    except KeyError:
        print(key + ' isn\'t defined')
        raise KeyError
    
    except InvalidSettings:
        print(key + ' sould be in', val_list)
        raise InvalidSettings
    else:
        print(key + ' is Ok\n')

items2set = [('mode',   mode_list,   ['direct', 'from pi and psi']),
             ('lam_q',  lam_q_list,  ['fixed', 'recomputed']),
             ('mu',     duals_list,  ['fixed', 'zeros fixed','recomputed']),
             ('sigma',  duals_list,  ['fixed', 'zeros fixed','recomputed']),
             ('pi_psi', pi_psi_list, ['zeros fixed','recomputed']),
            ]

def set_options():
    options = {}
    for item in items2set:
        if item[0] == 'pi_psi' and options['mode'] == separate_mode:
            break
        print('Set %s' % item[0])
        for i in list(zip(item[1], item[2])):
            print(i)
        options[item[0]] = int(input())
        key_val_is_in(options, item[0], item[1])
    if options['mode'] == simultaneous_mode:
        options['rho'] = fixed
    clear_output()
    return options

algs = [LS, NNLS, PQ_R, BoundedLS, linprog]
algs_discr = ['LS with no constraints', 
              'LS with x => 0', 
              'LS with all non negatives except lambdas', 
              'LS with l <= x <= u',
              'LinProg']

def set_alg():
    print('Set algorithm:')
    for i in list(zip(algs, algs_discr)):
        print(i)
    alg = int(input())
    clear_output()
    return alg

all_forming = 0
non_cr_only = 1

idx_discr = ['Lambdas for all forming generators will be fixed', 'Lambdas for non crininals only will be fixed']

def set_idx_mode():
    print('Forming genetators\' indexes mode:')
    for i in list(zip([all_forming, non_cr_only], idx_discr)):
        print(i)
    idx_mode = int(input())
    clear_output()
    return idx_mode


def create_all_pos_settings():
    set_list = []
    c = list(itertools.product(mode_list, lam_q_list, 
                               duals_list, duals_list, 
                               pi_psi_list, algs))
    for i in c:
        tmp_dict = {}
        tmp_dict['mode'] = i[0]
        tmp_dict['lam_q'] = i[1]
        tmp_dict['mu'] = i[2]
        tmp_dict['sigma'] = i[3]
        tmp_dict['pi_psi'] = i[4]
        tmp_dict['rho'] = fixed
        set_list.append((tmp_dict, i[-1]))
    return set_list


def check_opt_alg(opt, alg):
    #if alg == PQ_R and opt['mode'] and not opt['lam_q']:
    #    print('All x shold be => 0, NNLS will be used')
    #    alg = NNLS
    if alg == BoundedLS and opt['mode']:
        print('BoundedLS not for mode = 1')
        if opt['lam_q']:
            alg = PQ_R
        else:
            alg = NNLS

    return alg