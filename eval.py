#!/usr/bin/env python
# Evaluation

###################################################################################                   
# TILE GROUP EXPERIMENTS
###################################################################################                   

from __future__ import print_function
import os
import subprocess
import argparse
import pprint
import numpy
import sys
import re
import logging
import fnmatch
import string
import argparse
import pylab
import datetime
import math

import numpy as np
import matplotlib.pyplot as plot

from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogLocator
from pprint import pprint, pformat
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
from operator import add

import csv
import brewer2mpl
import matplotlib

from options import *
from functools import wraps       

###################################################################################                   
# LOGGING CONFIGURATION
###################################################################################                   

LOG = logging.getLogger(__name__)
LOG_handler = logging.StreamHandler()
LOG_formatter = logging.Formatter(
    fmt='%(asctime)s [%(funcName)s:%(lineno)03d] %(levelname)-5s: %(message)s',
    datefmt='%m-%d-%Y %H:%M:%S'
)
LOG_handler.setFormatter(LOG_formatter)
LOG.addHandler(LOG_handler)
LOG.setLevel(logging.INFO)

###################################################################################                   
# OUTPUT CONFIGURATION
###################################################################################                   

BASE_DIR = os.path.dirname(__file__)
OPT_FONT_NAME = 'Helvetica'
OPT_GRAPH_HEIGHT = 300
OPT_GRAPH_WIDTH = 400

#COLOR_MAP = ('#F15854', '#9C9F84', '#F7DCB4', '#991809', '#5C755E', '#A97D5D')
COLOR_MAP = ( '#F58A87', '#80CA86', '#9EC9E9', "#F15854", "#66A26B", "#5DA5DA")
OPT_COLORS = COLOR_MAP

OPT_GRID_COLOR = 'gray'
OPT_LEGEND_SHADOW = False
OPT_MARKERS = (['o', 's', 'v', "^", "h", "v", ">", "x", "d", "<", "|", "", "|", "_"])
OPT_PATTERNS = ([ "////", "////", "o", "o", "\\\\" , "\\\\" , "//////", "//////", ".", "." , "\\\\\\" , "\\\\\\" ])

OPT_LABEL_WEIGHT = 'bold'
OPT_LINE_COLORS = ('#fdc086', '#b3e2cd', '#fc8d62', '#a6cee3', '#e41a1c')
OPT_LINE_WIDTH = 6.0
OPT_MARKER_SIZE = 10.0
DATA_LABELS = []

OPT_STACK_COLORS = ('#AFAFAF', '#F15854', '#5DA5DA', '#60BD68',  '#B276B2', '#DECF3F', '#F17CB0', '#B2912F', '#FAA43A')

# SET FONT

LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14
TINY_FONT_SIZE = 8

AXIS_LINEWIDTH = 1.3
BAR_LINEWIDTH = 1.2

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{euler}']

LABEL_FP = FontProperties(family=OPT_FONT_NAME, style='normal', size=LABEL_FONT_SIZE, weight='bold')
TICK_FP = FontProperties(family=OPT_FONT_NAME, style='normal', size=TICK_FONT_SIZE)
TINY_FP = FontProperties(family=OPT_FONT_NAME, style='normal', size=TINY_FONT_SIZE)

###################################################################################                   
# CONFIGURATION
###################################################################################                   

PERF_LOCAL = "/usr/bin/perf"
PERF = "/usr/lib/linux-tools/3.11.0-12-generic/perf"

OPERATORS = ("direct", "aggregate", "arithmetic")
SELECTIVITY = (0.2, 0.4, 0.6, 0.8, 1.0)
PROJECTIVITY = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
LAYOUTS = ("row", "column", "hybrid")

LOG_SELECTIVITY = (0.01, 0.1, 0.5, 1.0)


PROJECTIVITY_DIR = "../results/ycsb/performance/"

LABELS = ("InP", "CoW", "Log", "NVM-InP", "NVM-CoW", "NVM-Log")

LOG_NAME = "tile_group.log"

###################################################################################                   
# UTILS
###################################################################################                   

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def loadDataFile(n_rows, n_cols, path):
    file = open(path, "r")
    reader = csv.reader(file)
    
    data = [[0 for x in xrange(n_cols)] for y in xrange(n_rows)]
    
    row_num = 0
    for row in reader:
        column_num = 0
        for col in row:
            data[row_num][column_num] = float(col)
            column_num += 1
        row_num += 1
                
    return data

# # MAKE GRID
def makeGrid(ax):
    axes = ax.get_axes()
    axes.yaxis.grid(True, color=OPT_GRID_COLOR)
    for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(AXIS_LINEWIDTH)
    ax.set_axisbelow(True)

# # SAVE GRAPH
def saveGraph(fig, output, width, height):
    size = fig.get_size_inches()
    dpi = fig.get_dpi()
    LOG.debug("Current Size Inches: %s, DPI: %d" % (str(size), dpi))

    new_size = (width / float(dpi), height / float(dpi))
    fig.set_size_inches(new_size)
    new_size = fig.get_size_inches()
    new_dpi = fig.get_dpi()
    LOG.debug("New Size Inches: %s, DPI: %d" % (str(new_size), new_dpi))
    
    pp = PdfPages(output)
    fig.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()
    LOG.info("OUTPUT: %s", output)

###################################################################################                   
# PLOT
###################################################################################                   

def create_legend():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(11, 0.5))

    num_items = len(ENGINES);   
    ind = np.arange(1)  
    margin = 0.10
    width = (1.0 - 2 * margin) / num_items      
      
    bars = [None] * len(LABELS) * 2

    for group in xrange(len(ENGINES)):        
        data = [1]
        bars[group] = ax1.bar(ind + margin + (group * width), data, width, 
                              color=OPT_COLORS[group], 
                              hatch=OPT_PATTERNS[group * 2], 
                              linewidth=BAR_LINEWIDTH)
        
    # LEGEND
    figlegend.legend(bars, LABELS, prop=LABEL_FP, loc=1, ncol=6, mode="expand", shadow=OPT_LEGEND_SHADOW, 
                     frameon=False, borderaxespad=0.0, handleheight=2, handlelength=3.5)

    figlegend.savefig('legend.pdf')

def create_projectivity_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)
         
    x_values = YCSB_SKEW_FACTORS
    N = len(x_values)
    x_labels = ["Low Skew", "High Skew"]

    num_items = len(ENGINES);   
    ind = np.arange(N)  
    margin = 0.10
    width = (1.0 - 2 * margin) / num_items      
    bars = [None] * len(LABELS) * 2

    YLIMIT = 2000000

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            if height > YLIMIT:
                label = '%.1f'%(height/1000000) + 'M'
                ax1.text(rect.get_x()+rect.get_width()/2., 1.05*YLIMIT, label,
                        ha='center', va='bottom', fontproperties=TINY_FP)

    # GROUP
    for group in xrange(len(datasets)):
        perf_data = []               

        # LINE
        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    perf_data.append(datasets[group][line][col])
  
        LOG.info("%s perf_data = %s ", LABELS[group], str(perf_data))
        
        bars[group] = ax1.bar(ind + margin + (group * width), perf_data, width, color=OPT_COLORS[group], hatch=OPT_PATTERNS[group * 2])        
        autolabel(bars[group])

    # RATIO
    transposed_datasets = map(list,map(None,*datasets))
    for type in xrange(N):
        LOG.info("type = %f ", x_values[type])
        get_ratio(transposed_datasets[type], True)
        
    # GRID
    axes = ax1.get_axes()
    makeGrid(ax1)
      
    # Y-AXIS
    ax1.yaxis.set_major_locator(MaxNLocator(5))
    ax1.minorticks_on()
    ax1.set_ylim([0,YLIMIT])
        
    # X-AXIS
    ax1.minorticks_on()
    ax1.set_xticklabels(x_labels)
    ax1.set_xticks(ind + 0.5)              
    ax1.set_ylabel("Throughput (txn/sec)", fontproperties=LABEL_FP)
    ax1.tick_params(axis='x', which='both', bottom='off', top='off')
        
    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)
            
    return (fig)

###################################################################################                   
# PLOT HELPERS                  
###################################################################################                   

# PROJECTIVITY -- PLOT
def projectivity_plot(result_dir, latency_list, prefix):
    for workload in YCSB_WORKLOAD_MIX:    

        for lat in latency_list:
            datasets = []
        
            for sy in SYSTEMS:    
                dataFile = loadDataFile(2, 2, os.path.realpath(os.path.join(result_dir, sy + "/" + workload + "/" + lat + "/performance.csv")))
                datasets.append(dataFile)
                                   
            fig = create_ycsb_perf_bar_chart(datasets)
            
            fileName = prefix + "ycsb-perf-%s-%s.pdf" % (workload, lat)
            saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/1.5)
           
###################################################################################                   
# EVAL                   
###################################################################################

# PROJECTIVITY -- EVAL
def projectivity_eval():

    nvm_latencies = latency_list
    rw_mixes = YCSB_RW_MIXES
    skew_factors = YCSB_SKEW_FACTORS
    engines = ENGINES
    
    # LOG RESULTS
    log_file = open(LOG_NAME, 'w')
    log_file.write('Start :: %s \n' % datetime.datetime.now())
    
    for layout in LAYOUTS:

        ostr = ("LAYOUT %s \n" % layout)  
        print (ostr, end="")
        log_file.write(ostr)
        log_file.flush()
        
        # START PG
        start_pg()
                           
        # EXPERIMENTS        
        for op in OPERATOR:

            for proj in PROJECTIVITY:

                ostr = ("--------------------------------------------------- \n")
                print (ostr, end="")
                log_file.write(ostr)
                
                ostr = ("TRIAL :: %d LAYOUT :: %s OP :: %s PROJ :: %.1f \n" % (layout, op, proj))
                print (ostr, end="")
                log_file.write(ostr)                    
                log_file.flush()
                
                # DO SOMETHING               
                #cleanup(log_file)
                #subprocess.call([NUMACTL, NUMACTL_FLAGS, NSTORE, '-k', str(keys), '-x', str(txns), '-p', str(rw_mix), '-q', str(skew_factor), eng], stdout=log_file)
                
                         
    # FINISH LOG
    log_file.write('End :: %s \n' % datetime.datetime.now())
    log_file.close()   
    log_file = open(log_name, "r")

###################################################################################                   
# MAIN
###################################################################################
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run Tilegroup Experiments')
    
    parser.add_argument("-p", "--projectivity", help='eval projectivity', action='store_true')
    parser.add_argument("-s", "--selectivity", help='eval selectivity', action='store_true')
    parser.add_argument("-o", "--operator", help='eval operator', action='store_true')
    
    args = parser.parse_args()
                    
    if args.projectivity:
        projectivity_eval()
    
