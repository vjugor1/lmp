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

import pypower.idx_gen as idx_gen
import pypower.idx_bus as idx_bus
import pypower.idx_brch as idx_brch
import pypower.idx_cost as idx_cost


def draw_net(net, results=None):
    scale_size=True

    bus_size, ext_grid_size, switch_size = 1. , 1., 1.
    switch_distance, load_size, sgen_size, trafo_size = 1. , 1., 1., 1.

    if scale_size:
        mean_distance_between_buses = sum((net['bus_geodata'].max() - net[
            'bus_geodata'].min()).dropna() / 200)
        bus_size *= mean_distance_between_buses
        ext_grid_size *= mean_distance_between_buses * 1.5
        switch_size *= mean_distance_between_buses * 1
        switch_distance *= mean_distance_between_buses * 2
        load_size *= mean_distance_between_buses
        sgen_size *= mean_distance_between_buses
        trafo_size *= mean_distance_between_buses
    fp = FontProperties(family="Helvetica", style="italic")
    bc = create_bus_collection(net,
                               net.bus.index,
                               size=bus_size,
                               bus_geodata=net.bus_geodata)


    use_bus_geodata = len(net.line_geodata) == 0
    in_service_lines = net.line[net.line.in_service].index
    respect_switches=False
    nogolines = set(net.switch.element[(net.switch.et == "l") & (net.switch.closed == 0)]) \
        if respect_switches else set()
    plot_lines = in_service_lines.difference(nogolines)

    lc = create_line_collection(net,
                                plot_lines,
                                use_bus_geodata=use_bus_geodata)

    buses = net.bus.index.tolist()
    
    coords = zip(net.bus_geodata.loc[buses, "x"].values, net.bus_geodata.loc[buses, "y"].values)
    if results != None:
        lambs = [round(results['bus'][idx, LAM_P],2) for idx in net.bus.index.tolist()]
        ac    = create_annotation_collection(['  '+ str(b + 1) + ' lam='+str(l) for b, l in zip(buses, lambs)], coords, 0.1, prop=fp)
    else:
        ac    = create_annotation_collection([str(b + 1) for b in buses], coords, 0.2)

    
    gc = create_gen_collection(net, size=sgen_size*2)

    ec = create_ext_grid_collection(net, size=sgen_size*2)
    
    collections = [bc, lc, ac, gc, ec]
    draw_collections(collections)


def draw_nasty_profile(results, ylimit=None):
    plt.figure(figsize=(10, 5))
    plt.plot(results['bus'][:, 0], results['new_lam'], label='new')
    plt.plot(results['bus'][:, 0], results['bus'][:, LAM_P], label='old')
    plt.ylabel('LAM_P')
    plt.xlabel('BUS#')
    if ylimit is not None:
        plt.ylim(ylimit)
    plt.legend(loc='best')
    
    
def draw_nice_hists(gens, results):
    loads = np.setdiff1d(results['gen'][:, 0], gens)

    ind = np.arange(len(gens))  # the x locations for the groups
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    rects1 = ax[0].bar(ind - width, results['bus'][:, LAM_P][gens.astype(int) - 1], width,
                    label='Old')
    rects2 = ax[0].bar(ind + width, results['new_lam'][gens.astype(int) - 1], width,
                    label='New')
    #print('ind = ', ind)
    #print('results[gencost][:, 4][results[gencost][:, 4] > 0] = ', results['gencost'][:, 4][results['gencost'][:, 4] > 0])
    rects2 = ax[0].bar(ind, results['gencost'][:, 4][results['gencost'][:, 4] > 0], width,
                    label='Cost')

    ax[0].set_ylabel('LAM_P')
    ax[0].set_title('Generation lambdas')
    ax[0].set_xticks(ind)
    ax[0].set_xticklabels(('G#%d' % i for i in gens))
    ax[0].legend()

    ind = np.arange(len(loads))
    rects1 = ax[1].bar(ind - width, results['bus'][:, LAM_P][loads.astype(int) - 1], width,
                    label='Old')
    rects2 = ax[1].bar(ind + width, results['new_lam'][loads.astype(int) - 1], width,
                    label='New')

    rects2 = ax[1].bar(ind, -results['gencost'][:, 4][~(results['gencost'][:, 4] > 0)], width,
                    label='Cost')

    ax[1].set_ylabel('LAM_P')
    ax[1].set_title('Demand lambdas')
    ax[1].set_xticks(ind)
    ax[1].set_xticklabels(('L#%d' % i for i in loads))
    ax[1].legend()
