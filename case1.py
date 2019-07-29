import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
#%matplotlib inline

from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PQ, REF
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS
from pypower.idx_bus import BUS_TYPE, REF, VM, VA, MU_VMAX, MU_VMIN, LAM_P, LAM_Q
from pypower.idx_brch import F_BUS, T_BUS, RATE_A, RATE_B, PF, QF, PT, QT, MU_SF, MU_ST
from pypower.idx_gen import GEN_BUS, PG, QG, VG, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN
from pypower.idx_cost import MODEL, PW_LINEAR, NCOST

from pypower.api import makeYbus
from pypower.api import case30pwl, case30Q 
from pypower.api import case30, ppoption, runpf, printpf, runopf
from pypower.t import t_loadcase, t_auction_case
from pypower.ext2int import ext2int
from scipy.optimize import minimize, nnls, linprog


import pandapower.converter as pc
from pandapower.networks import case30 as case30_panda

from drawings import draw_net

from utils import *

from solution_options import set_options, set_alg, check_opt_alg, set_idx_mode

from solver import *
from utils import Rand_replace, Median_func

from drawings import draw_nice_hists, draw_nasty_profile


from pandapower.networks import case30 as case30_panda


class case1:
	def __init__(self):
		self.ppc = case30()
		self.ppopt = ppoption(PF_ALG=1, RETURN_RAW_DER=True, OPF_FLOW_LIM=0)
		self.ppn = case30_panda()
		self.results = None
		self.sol_opt = None
		self.alg = None
		self.idx_mode = None
		self.res_struct = None
		self.load_list = None
	def set_bids(self, bid_list=[7., 15., 30.]):
		gens = self.ppc['gen'][:, 0]

		self.load_list = []

		for bus_str in self.ppc['bus']:
			tmp = np.zeros(21)
			if bus_str[0] in bid_list:
				if bus_str[0] not in gens:
					#print('load #%d' % bus_str[0])
					tmp[0] = bus_str[0]

					tmp[1] = -30.#bus_str[2]
					tmp[2] = -15.#bus_str[3]	
					tmp[3] = -0.0001
					tmp[4] = -15.

					tmp[5] = bus_str[7]
					tmp[6] = 100.

					tmp[7] = 1.

					#fixed here
					tmp[8] = -0.0001
					tmp[9] = -30.

					bus_str[2], bus_str[3] = 0., 0.
					self.load_list.append(tmp)

		self.ppc['gen'] = np.concatenate((self.ppc['gen'], np.array(self.load_list)))


	def fasten_branches(self, alpha=1.0, branches=[(14,5), (25, 5), (31, 5)]):
		"""
		Input:
		ppc: ppc PyPower case struct
		alpha: factor to fasten the branches
		bracnhes is a list of tuples
		"""
		#bound max flow 3 <-> 11
		for tuple_ in branches:
			from_ = tuple_[0]
			to_   = tuple_[1]
			try:
				self.ppc['branch'][from_][to_] *= alpha
			except ValueError:
				print('Branch (', from_, ', ', 'to_', 'does not exist') 

	def set_costs(self, costs_list = [1000, 1000, 1100, 1200, 10000, 13000, -40000, -40000, -40000]):
		self.ppc['gencost'][:,4] = costs_list

	def prepare_gen_cost(self, costs_list = [1000, 1000, 1100, 1200, 10000, 13000, -40000, -40000, -40000]):
		self.ppc['gencost'][:,3] = np.ones(self.ppc['gencost'].shape[0]) * 2
		self.ppc['gencost'] = self.ppc['gencost'][:,[0,1,2,3,5,6]]
		cost_list = []

		for load in self.load_list:
    			cost_list.append(np.array([2., 0., 0., 2., -10., 0]))
    
		self.ppc['gencost'] = np.concatenate((self.ppc['gencost'], np.array(cost_list)))
		
		self.set_costs(costs_list)		

	def draw_ppn_net(self):
		draw_net(self.ppn)
	def runopf_case(self):
		self.results = runopf(self.ppc, self.ppopt)

	def get_price_formings(self, verbose=True):
		if self.results == None:
			raise ValueError('results==None. Use runopf_case method first')
		else:
			return is_price_forming(self.results['gen'], self.results['gencost'], verbose=verbose)

	def get_jac(self):
		if self.results == None:
			raise ValueError('results==None. Use runopf_case method first')
		else:
			return get_Jac(self.ppc, self.ppopt, self.results)
	def set_opts_alg(self, threshold=2000.):
		self.sol_opt = set_options()
		self.alg = set_alg()
		self.idx_mode = set_idx_mode()
		self.alg = check_opt_alg(self.sol_opt, self.alg)

	

	def solve(self, threshold=2000., alpha=0.2, beta=0.8):
		if self.results == None:
			raise ValueError('results==None. Use runopf_case method first')
		else:
			self.set_opts_alg(threshold)
			dgT, dgeqT = self.get_jac()
			self.res_struct = Solve(dgT, dgeqT, self.results, threshold, 
				                   replace_func=Rand_replace, 
				                   opt=self.sol_opt,
				                   alg=self.alg, 
				                   idx_mode=self.idx_mode,
				                   alpha=alpha, beta=beta)
	def prepare_case(self, bid_list=[7.0, 15.0, 30.0], alpha=1.0, branches=[(14, 5), (25, 5), (31, 5)], costs_list = [1000, 1000, 1100, 1200, 10000, 13000, -40000, -40000, -40000]):
		self.set_bids(bid_list)
		self.fasten_branches(alpha, branches)
		self.prepare_gen_cost()

	def plot_profile(self, ylim_low, ylim_high):
		draw_nasty_profile(self.results, (ylim_low, ylim_high))
	def plot_hists(self):
		gens = self.results['gen'][:, 0][self.results['gencost'][:,4] > 0]
		draw_nice_hists(gens, self.results)
	def plot(self, ylim_low, ylim_high):
		gens = self.results['gen'][:, 0][self.results['gencost'][:,4] > 0]
		
		draw_nasty_profile(self.results, (ylim_low, ylim_high))
		draw_nice_hists(gens, self.results)









