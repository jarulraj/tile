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
OPT_LINE_WIDTH = 4.0
OPT_MARKER_SIZE = 6.0
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

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

PELOTON_BUILD_DIR = BASE_DIR + "/../peloton/build"
HYADAPT = PELOTON_BUILD_DIR + "/src/hyadapt"

OUTPUT_FILE = "outputfile.summary"

PROJECTIVITY_DIR = BASE_DIR + "/results/projectivity/"
SELECTIVITY_DIR = BASE_DIR + "/results/selectivity/"

LAYOUTS = ("row", "column", "hybrid")
OPERATORS = ("direct", "aggregate", "arithmetic")

SELECTIVITY = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
PROJECTIVITY = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

SCALE_FACTOR = 1000.0
TRANSACTION_COUNT = 3

PROJECTIVITY_EXPERIMENT = 1
SELECTIVITY_EXPERIMENT = 1

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

    figlegend = pylab.figure(figsize=(6, 0.5))
    idx = 0
    lines = [None] * len(LAYOUTS)

    layouts = ("Row", "Column", "Hybrid")

    for group in xrange(len(LAYOUTS)):
        data = [1]
        x_values = [1]

        lines[idx], = ax1.plot(x_values, data, color=OPT_LINE_COLORS[idx], linewidth=OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx], markersize=OPT_MARKER_SIZE, label=str(group))

        idx = idx + 1

    # LEGEND
    figlegend.legend(lines,  layouts, prop=LABEL_FP, loc=1, ncol=4, mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0, handleheight=2, handlelength=3.5)

    figlegend.savefig('legend.pdf')


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
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    #ax1.set_ylim([0, YLIMIT])

    # X-AXIS
    ax1.set_xlabel("Fraction of Attributes Projected", fontproperties=LABEL_FP)
    ax1.set_xlim([0.0, 1.1])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_selectivity_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = SELECTIVITY
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
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    #ax1.set_ylim([0, YLIMIT])

    # X-AXIS
    ax1.set_xlabel("Fraction of Tuples Selected", fontproperties=LABEL_FP)
    ax1.set_xlim([0.0, 1.1])

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

# SELECTIVITY -- PLOT
def selectivity_plot():

    for operator in OPERATORS:
        print(operator)
        datasets = []

        for layout in LAYOUTS:
            data_file = SELECTIVITY_DIR + "/" + layout + "/" + operator + "/" + "selectivity.csv"

            dataset = loadDataFile(10, 2, data_file)
            datasets.append(dataset)

        fig = create_selectivity_line_chart(datasets)

        fileName = "selectivity-%s.pdf" % (operator)
        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/1.5)

###################################################################################
# EVAL HELPERS
###################################################################################

# CLEAN UP RESULT DIR
def clean_up_dir(result_directory):

    subprocess.call(['rm', '-rf', result_directory])
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

# RUN EXPERIMENT
def run_experiment(experiment_type):

    # cleanup
    subprocess.call(["rm -f " + OUTPUT_FILE], shell=True)

    subprocess.call([HYADAPT,
                     "-e", str(experiment_type),
                     "-k", str(SCALE_FACTOR),
                     "-t", str(TRANSACTION_COUNT)])


# COLLECT STATS
def collect_stats(result_dir,
                  result_file_name):

    fp = open(OUTPUT_FILE)
    lines = fp.readlines()
    fp.close()

    for line in lines:
        data = line.split()

        # Collect info
        layout = data[0]
        operator = data[1]
        selectivity = data[2]
        projectivity = data[3]
        stat = data[4]

        if(layout == "0"):
            layout = "row"
        elif(layout == "1"):
            layout = "column"
        elif(layout == "2"):
            layout = "hybrid"

        if(operator == "1"):
            operator = "direct"
        elif(operator == "2"):
            operator = "aggregate"
        elif(operator == "3"):
            operator = "arithmetic"

        result_directory = result_dir + "/" + layout + "/" + operator

        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        file_name = result_directory + "/" + result_file_name

        result_file = open(file_name, "a")
        result_file.write(str(projectivity) + " , " + str(stat) + "\n")
        result_file.close()

###################################################################################
# EVAL
###################################################################################

# PROJECTIVITY -- EVAL
def projectivity_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(PROJECTIVITY_DIR)

    # RUN EXPERIMENT
    run_experiment(PROJECTIVITY_EXPERIMENT)

    # COLLECT STATS
    collect_stats(PROJECTIVITY_DIR, "projectivity.csv")

# SELECTIVITY -- EVAL
def selectivity_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(SELECTIVITY_DIR)

    # RUN EXPERIMENT
    run_experiment(SELECTIVITY_EXPERIMENT)

    # COLLECT STATS
    collect_stats(SELECTIVITY_DIR, "selectivity.csv")

###################################################################################
# MAIN
###################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Tilegroup Experiments')

    parser.add_argument("-p", "--projectivity", help='eval projectivity', action='store_true')
    parser.add_argument("-s", "--selectivity", help='eval selectivity', action='store_true')
    parser.add_argument("-o", "--operator", help='eval operator', action='store_true')

    parser.add_argument("-a", "--projectivity_plot", help='plot projectivity', action='store_true')
    parser.add_argument("-b", "--selectivity_plot", help='plot selectivity', action='store_true')

    args = parser.parse_args()

    if args.projectivity:
        projectivity_eval()

    if args.projectivity_plot:
       projectivity_plot();

    if args.selectivity:
        selectivity_eval()

    if args.selectivity_plot:
       selectivity_plot();

    #create_legend()


