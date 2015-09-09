#!/usr/bin/env python
# Evaluation

# # TILE GROUP EXPERIMENTS

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


# # LOGGING CONFIGURATION
LOG = logging.getLogger(__name__)
LOG_handler = logging.StreamHandler()
LOG_formatter = logging.Formatter(
    fmt='%(asctime)s [%(funcName)s:%(lineno)03d] %(levelname)-5s: %(message)s',
    datefmt='%m-%d-%Y %H:%M:%S'
)
LOG_handler.setFormatter(LOG_formatter)
LOG.addHandler(LOG_handler)
LOG.setLevel(logging.INFO)


# # CONFIGURATION

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

# # TILE GROUP
PERF_LOCAL = "/usr/bin/perf"
PERF = "/usr/lib/linux-tools/3.11.0-12-generic/perf"
NUMACTL = "numactl"
NUMACTL_FLAGS = "--membind=2"

SYSTEMS = ("wal", "sp", "lsm", "opt_wal", "opt_sp", "opt_lsm")
RECOVERY_SYSTEMS = ("wal", "lsm", "opt_wal", "opt_lsm")
LATENCIES = ("160", "320", "1280")

ENGINES = ['-a', '-s', '-m', '-w', '-c', '-l']

YCSB_KEYS = 2000000
YCSB_TXNS = 8000000
YCSB_WORKLOAD_MIX = ("read-only", "read-heavy", "balanced", "write-heavy")

YCSB_PERF_DIR = "../results/ycsb/performance/"

LABELS = ("InP", "CoW", "Log", "NVM-InP", "NVM-CoW", "NVM-Log")

TPCC_TXNS = 1000000
TEST_TXNS = 500000

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
        bars[group] = ax1.bar(ind + margin + (group * width), data, width, color=OPT_COLORS[group], hatch=OPT_PATTERNS[group * 2], linewidth=BAR_LINEWIDTH)
        
    # LEGEND
    figlegend.legend(bars, LABELS, prop=LABEL_FP, loc=1, ncol=6, mode="expand", shadow=OPT_LEGEND_SHADOW, 
                     frameon=False, borderaxespad=0.0, handleheight=2, handlelength=3.5)

    figlegend.savefig('legend.pdf')

def create_ycsb_perf_bar_chart(datasets):
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


# YCSB PERF -- PLOT
def ycsb_perf_plot(result_dir, latency_list, prefix):
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

# CLEANUP PMFS

SDV_DEVEL = "/data/devel/sdv-tools/"
FS_ABS_PATH = "/mnt/pmfs/"

def cleanup(log_file):
    # LOCAL
    if enable_local:        
        subprocess.call(["rm -f " + FS_PATH + "./*"], shell=True)    
           
    # PMFS            
    else:
        cwd = os.getcwd()
        os.chdir(SDV_DEVEL)
        subprocess.call(['sudo', 'umount', '/mnt/pmfs'], stdout=log_file)
        subprocess.call(['sudo', 'bash', 'mount_pmfs.sh'], stdout=log_file)        
        os.chdir(FS_ABS_PATH)
        subprocess.call(['sudo', 'mkdir', 'n-store'], stdout=log_file)
        subprocess.call(['sudo', 'chown', 'user', 'n-store'], stdout=log_file)
        os.chdir(cwd)    


# YCSB PERF -- EVAL
def ycsb_perf_eval(enable_sdv, enable_trials, log_name, result_dir, latency_list):        
    dram_latency = 100
    keys = YCSB_KEYS
    txns = YCSB_TXNS
                            
    num_trials = 1 
    if enable_trials: 
        num_trials = 3
    
    nvm_latencies = latency_list
    rw_mixes = YCSB_RW_MIXES
    skew_factors = YCSB_SKEW_FACTORS
    engines = ENGINES
    
    # LOG RESULTS
    log_file = open(log_name, 'w')
    log_file.write('Start :: %s \n' % datetime.datetime.now())
    
    for nvm_latency in nvm_latencies:

        ostr = ("LATENCY %s \n" % nvm_latency)    
        print (ostr, end="")
        log_file.write(ostr)
        log_file.flush()
        
        if enable_sdv :
            cwd = os.getcwd()
            os.chdir(SDV_DIR)
            subprocess.call(['sudo', SDV_SCRIPT, '--enable', '--pm-latency', str(nvm_latency)], stdout=log_file)
            os.chdir(cwd)
                   
        for trial in range(num_trials):
            # RW MIX
            for rw_mix  in rw_mixes:
                # SKEW FACTOR
                for skew_factor  in skew_factors:
                    ostr = ("--------------------------------------------------- \n")
                    print (ostr, end="")
                    log_file.write(ostr)
                    ostr = ("TRIAL :: %d RW MIX :: %.1f SKEW :: %.2f \n" % (trial, rw_mix, skew_factor))
                    print (ostr, end="")
                    log_file.write(ostr)                    
                    log_file.flush()
                               
                    for eng in engines:
                        cleanup(log_file)
                        subprocess.call([NUMACTL, NUMACTL_FLAGS, NSTORE, '-k', str(keys), '-x', str(txns), '-p', str(rw_mix), '-q', str(skew_factor), eng], stdout=log_file)

    # RESET
    if enable_sdv :
        cwd = os.getcwd()
        os.chdir(SDV_DIR)
        subprocess.call(['sudo', SDV_SCRIPT, '--enable', '--pm-latency', "200"], stdout=log_file)
        os.chdir(cwd)
 
    # PARSE LOG
    log_file.write('End :: %s \n' % datetime.datetime.now())
    log_file.close()   
    log_file = open(log_name, "r")    

    tput = {}
    mean = {}
    sdev = {}
    latency = 0
    rw_mix = 0.0
    skew = 0.0
    
    skew_factors = []
    nvm_latencies = []
    engine_types = []
    
    for line in log_file:
        if "LATENCY" in line:
            entry = line.strip().split(' ');
            if entry[0] == "LATENCY":
                latency = entry[1]
            if latency not in nvm_latencies:
                nvm_latencies.append(latency)
                    
        if "RW MIX" in line:
            entry = line.strip().split(' ');
            trial = entry[2]
            rw_mix = entry[6]
            skew = entry[9]
            
            if skew not in skew_factors:
                skew_factors.append(skew)
       
        if "Throughput" in line:
            entry = line.strip().split(':');
            engine_type = entry[0].split(' ');
            val = float(entry[4]);
            
            if(engine_type[0] == "WAL"):
                engine_type[0] = "wal"                
            elif(engine_type[0] == "SP"):
                engine_type[0] = "sp"
            elif(engine_type[0] == "LSM"):
                engine_type[0] = "lsm"
            elif(engine_type[0] == "OPT_WAL"):
                engine_type[0] = "opt_wal"
            elif(engine_type[0] == "OPT_SP"):
                engine_type[0] = "opt_sp"
            elif(engine_type[0] == "OPT_LSM"):
                engine_type[0] = "opt_lsm"
            
            if engine_type not in engine_types:
                engine_types.append(engine_type)
                            
            key = (rw_mix, skew, latency, engine_type[0]);
            if key in tput:
                tput[key].append(val)
            else:
                tput[key] = [ val ]
                            

    # CLEAN UP RESULT DIR
    subprocess.call(['rm', '-rf', result_dir])          
    
    for key in sorted(tput.keys()):
        mean[key] = round(numpy.mean(tput[key]), 2)
        mean[key] = str(mean[key]).rjust(10)
            
        sdev[key] = numpy.std(tput[key])
        sdev[key] /= float(mean[key])
        sdev[key] = round(sdev[key], 3)
        sdev[key] = str(sdev[key]).rjust(10)
        
        engine_type = str(key[3]);        
        if(key[0] == '0.0'):
            workload_type = 'read-only'
        elif(key[0] == '0.1'):
            workload_type = 'read-heavy'
        elif(key[0] == '0.5'):
            workload_type = 'balanced'
        elif(key[0] == '0.9'):
            workload_type = 'write-heavy'
    
        nvm_latency = str(key[2]);
        
        result_directory = result_dir + engine_type + "/" + workload_type + "/" + nvm_latency + "/";
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)

        result_file_name = result_directory + "performance.csv"
        result_file = open(result_file_name, "a")
        result_file.write(str(key[1] + " , " + mean[key] + "\n"))
        result_file.close()    
                    
    read_only = []
    read_heavy = []
    write_heavy = []
    
    # ARRANGE DATA INTO TABLES    
    for key in sorted(mean.keys()):
        if key[0] == '0.0':
            read_only.append(str(mean[key] + "\t" + sdev[key] + "\t"))
        elif key[0] == '0.1':
            read_heavy.append(str(mean[key] + "\t" + sdev[key] + "\t"))
        elif key[0] == '0.5':
            write_heavy.append(str(mean[key] + "\t" + sdev[key] + "\t"))
        
    col_len = len(nvm_latencies) * len(engine_types)           
        
    ro_chunks = list(chunks(read_only, col_len))
    print('\n'.join('\t'.join(map(str, row)) for row in zip(*ro_chunks)))
    print('\n', end="")
        
    rh_chunks = list(chunks(read_heavy, col_len))
    print('\n'.join('\t'.join(map(str, row)) for row in zip(*rh_chunks)))
    print('\n', end="")
        
    wh_chunks = list(chunks(write_heavy, col_len))
    print('\n'.join('\t'.join(map(str, row)) for row in zip(*wh_chunks)))
    print('\n', end="")

## ==============================================
# # main
## ==============================================
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run Tilegroup Experiments')
    
    parser.add_argument("-y", "--ycsb_perf_eval", help='eval ycsb perf', action='store_true')
    
    args = parser.parse_args()

    ycsb_perf_log_name = "ycsb_perf.log"
            
    ################################ YCSB
    
    if args.ycsb_perf_eval:
        ycsb_perf_eval(enable_sdv, enable_trials, ycsb_perf_log_name, YCSB_PERF_DIR, LATENCIES)
    
