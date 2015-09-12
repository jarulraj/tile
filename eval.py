#!/usr/bin/env python

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
import time
import fileinput
from lxml import etree

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

PG_CTL = "/usr/local/peloton/bin/pg_ctl"
PG_DATA_DIR = "./data"
PG_BUILD_DIR = "../peloton/build"
OLTPBENCH_DIR = "../oltpbench"
PG_CONFIG_FILE = PG_DATA_DIR + "/postgresql.conf"

OLTPBENCH = "./oltpbenchmark"

BENCHMARK_NAME = "hyadapt"
CONFIG_FILE = "config/peloton_hyadapt_config.xml"
OUTPUT_FILE = "outputfile"

OPERATORS = ("direct", "aggregate", "arithmetic")
SELECTIVITY = (0.2, 0.4, 0.6, 0.8, 1.0)
PROJECTIVITY = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
LAYOUTS = ("row", "column", "hybrid")

LOG_SELECTIVITY = (0.01, 0.1, 0.5, 1.0)
SCALE_FACTOR = 1.0
TIME = 10.0

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECTIVITY_DIR = BASE_DIR + "/results/projectivity/"

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

def create_projectivity_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)
         
    # X-AXIS
    x_values = PROJECTIVITY
    N = len(x_values)
    x_labels = x_values

    num_items = len(LAYOUTS);
    ind = np.arange(N)  
    idx = 0
    
    YLIMIT = 100
            
    # GROUP
    for group_index, group in enumerate(LAYOUTS):
        group_data = []             
        
        # LINE
        for line_index, line in enumerate(x_values):            
            group_data.append(datasets[group_index][line_index][1])
  
        LOG.info("%s group_data = %s ", group, str(group_data))
        
        ax1.plot(x_values, group_data, color=OPT_LINE_COLORS[idx], linewidth=OPT_LINE_WIDTH, 
                 marker=OPT_MARKERS[idx], markersize=OPT_MARKER_SIZE, label=str(group))        
        
        idx = idx + 1  

    # GRID
    axes = ax1.get_axes()
    makeGrid(ax1)
      
    # Y-AXIS    
    ax1.yaxis.set_major_locator(MaxNLocator(5))
    ax1.minorticks_on()
    ax1.set_ylabel("Throughput (txn/sec)", fontproperties=LABEL_FP)
    #ax1.set_ylim([0, YLIMIT])
        
    # X-AXIS
    ax1.minorticks_on()
    ax1.set_xlabel("Fraction of Attributes Projected", fontproperties=LABEL_FP)
        
    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)
            
    return (fig)

###################################################################################                   
# PLOT HELPERS                  
###################################################################################                   

# PROJECTIVITY -- PLOT
def projectivity_plot():
    
    for operator in OPERATORS:    
        print(operator)
        datasets = []

        for layout in LAYOUTS:        
            data_file = PROJECTIVITY_DIR + "/" + layout + "/" + operator + "/" + "projectivity.csv"
            
            dataset = loadDataFile(10, 2, data_file)
            datasets.append(dataset)

        fig = create_projectivity_line_chart(datasets)
        
        fileName = "projectivity-%s.pdf" % (operator)
        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/1.5)
           
###################################################################################                   
# EVAL HELPERS                   
###################################################################################

# CLEAN UP RESULT DIR
def clean_up_dir(result_directory):

    subprocess.call(['rm', '-rf', result_directory])
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

# UPDATE POSTGRES CONFIG FILE
def update_postgres_config_file(layout):
    
    text_to_search = "layout"
    text_to_replace = layout
    for line in fileinput.input(PG_CONFIG_FILE, inplace=True):

        # LAYOUT UPDATE        
        if text_to_search in line:
            line = line.replace("row", text_to_replace)
            line = line.replace("column", text_to_replace)
            line = line.replace("hybrid", text_to_replace)
            
# EXECUTE PG
def execute_pg(log_file, layout):
    
    cwd = os.getcwd()
    os.chdir(PG_BUILD_DIR)
    
    update_postgres_config_file(layout)
     
    subprocess.call([PG_CTL, '-D', PG_DATA_DIR, 'stop'], stdout=log_file)
    subprocess.call([PG_CTL, '-D', PG_DATA_DIR, 'start'], stdout=log_file)

    os.chdir(cwd)

# UPDATE OLTPBENCH CONFIG FILE
def update_oltpbench_config_file(operator, projectivity, selectivity):

    tree = etree.parse(CONFIG_FILE)
    
    root = tree.getroot()
    
    # SCALE FACTOR
    root.find('scalefactor').text = str(SCALE_FACTOR)
    
    # SELECTIVITY
    root.find('selectivity').text = str(selectivity)

    workload = [0] * 30

    # OPERATOR
    op_index = OPERATORS.index(operator)    

    # PROJECTIVITY
    proj_index = projectivity * 10 - 1
    
    if(op_index <= 2):
        workload[op_index * 10 + int(proj_index)] = 100

    # WORKLOAD
    works = root.find('works')
    work = works.find('work') 
       
    weights = work.find('weights')
    time = work.find('time')
    
    workload_string = ", ".join( repr(e) for e in workload)
    weights.text = workload_string
    print("WORKLOAD : " + workload_string)
    
    time.text = str(TIME)
        
    fp = open(CONFIG_FILE, 'w')
    fp.write(etree.tostring(root, pretty_print=True))
    fp.close()
            
# RUN OLTPBENCH
def run_oltpbenchmark(log_file):
    
    # cleanup     
    subprocess.call(["rm -f " + OUTPUT_FILE+".*"], shell=True)   

    # ./oltpbenchmark -b hyadapt -c config/peloton_hyadapt_config.xml 
    # --create=true --load=true --execute=true -s 5 -o outputfile     
    subprocess.call([OLTPBENCH, '-b', BENCHMARK_NAME, '-c', CONFIG_FILE,
                     '--create=true', '--load=true', '--execute=true',
                     '-s', '5', '-o', OUTPUT_FILE], stdout=log_file)

# COLLECT STATS    
def collect_stats(layout, operator, projectivity, selectivity, result_dir,
                  result_file_name):
    fp = open(OUTPUT_FILE + ".summary")
    lines = fp.readlines()
    
    # Collect info
    stat = lines[5].rstrip()
    
    result_directory = result_dir + "/" + layout + "/" + operator
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)  
    result_file_name = result_directory + "/" + result_file_name
    result_file = open(result_file_name, "a")
    result_file.write(str(projectivity) + " , " + str(stat) + "\n")
    result_file.close()    
    
    fp.close()

# EXECUTE OLTPBENCH
def execute_oltpbenchmark(log_file, layout, operator, projectivity, selectivity,
                          result_dir, result_file_name):
    cwd = os.getcwd()
    os.chdir(OLTPBENCH_DIR)

    # First, update the config file
    update_oltpbench_config_file(operator, projectivity, selectivity)
    
    # Second, run benchmark
    #run_oltpbenchmark(log_file)
        
    # Finally, collect stats
    collect_stats(layout, operator, projectivity, selectivity, 
                  result_dir, result_file_name)
    
    os.chdir(cwd)

###################################################################################                   
# EVAL                   
###################################################################################

# PROJECTIVITY -- EVAL
def projectivity_eval():

    selectivity = 1.0

    # LOG RESULTS
    log_file = open(LOG_NAME, 'w')
    log_file.write('Start :: %s \n' % datetime.datetime.now())
    
    # CLEAN UP RESULT DIR
    clean_up_dir(PROJECTIVITY_DIR)
    
    for layout in LAYOUTS:

        ostr = ("LAYOUT %s \n" % layout)  
        print (ostr, end="")
        log_file.write(ostr)
        log_file.flush()
        
        # EXECUTE PG
        execute_pg(log_file, layout)
                           
        # EXPERIMENTS        
        for operator in OPERATORS:

            ostr = ("--------------------------------------------------- \n")
            print (ostr, end="")
            log_file.write(ostr)

            for projectivity in PROJECTIVITY:
                
                ostr = ("LAYOUT :: %s OP :: %s PROJ :: %.1f \n" % (layout, operator, projectivity))
                print (ostr, end="")
                log_file.write(ostr)                    
                log_file.flush()
                                                
                # EXECUTE BENCHMARK
                execute_oltpbenchmark(log_file, layout, operator, projectivity, selectivity,
                                      PROJECTIVITY_DIR, "projectivity.csv")
                
                         
    # FINISH LOG
    log_file.write('End :: %s \n' % datetime.datetime.now())
    log_file.close()   
    log_file = open(LOG_NAME, "r")

###################################################################################                   
# MAIN
###################################################################################
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run Tilegroup Experiments')
    
    parser.add_argument("-p", "--projectivity", help='eval projectivity', action='store_true')
    parser.add_argument("-s", "--selectivity", help='eval selectivity', action='store_true')
    parser.add_argument("-o", "--operator", help='eval operator', action='store_true')
    
    parser.add_argument("-a", "--projectivity_plot", help='plot projectivity', action='store_true')
    
    args = parser.parse_args()
                    
    if args.projectivity:
        projectivity_eval()
    
    if args.projectivity_plot:                
       projectivity_plot();                          
