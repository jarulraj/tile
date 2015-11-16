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
from matplotlib.ticker import LinearLocator
from pprint import pprint, pformat
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
from operator import add
import matplotlib.font_manager as font_manager

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

# Make a list by cycling through the colors you care about
# to match the length of your data.

NUM_COLORS = 5
COLOR_MAP = ( '#F58A87', '#80CA86', '#9EC9E9', '#FED113', '#D89761' )


#COLOR_MAP = ('#F15854', '#9C9F84', '#F7DCB4', '#991809', '#5C755E', '#A97D5D')
OPT_COLORS = COLOR_MAP

OPT_GRID_COLOR = 'gray'
OPT_LEGEND_SHADOW = False
OPT_MARKERS = (['o', 's', 'v', "^", "h", "v", ">", "x", "d", "<", "|", "", "|", "_"])
OPT_PATTERNS = ([ "////", "////", "o", "o", "\\\\" , "\\\\" , "//////", "//////", ".", "." , "\\\\\\" , "\\\\\\" ])

OPT_LABEL_WEIGHT = 'bold'
OPT_LINE_COLORS = COLOR_MAP
OPT_LINE_WIDTH = 6.0
OPT_MARKER_SIZE = 10.0
DATA_LABELS = []

OPT_STACK_COLORS = ('#AFAFAF', '#F15854', '#5DA5DA', '#60BD68',  '#B276B2', '#DECF3F', '#F17CB0', '#B2912F', '#FAA43A')

# SET FONT

LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
TINY_FONT_SIZE = 8
LEGEND_FONT_SIZE = 16

AXIS_LINEWIDTH = 1.3
BAR_LINEWIDTH = 1.2

# SET TYPE1 FONTS
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['font.family'] = OPT_FONT_NAME
matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{euler}']

LABEL_FP = FontProperties(style='normal', size=LABEL_FONT_SIZE, weight='bold')
TICK_FP = FontProperties(style='normal', size=TICK_FONT_SIZE)
TINY_FP = FontProperties(style='normal', size=TINY_FONT_SIZE)
LEGEND_FP = FontProperties(style='normal', size=LEGEND_FONT_SIZE, weight='bold')

YAXIS_TICKS = 3
YAXIS_ROUND = 1000.0

###################################################################################
# CONFIGURATION
###################################################################################

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

PELOTON_BUILD_DIR = BASE_DIR + "/../peloton/build"
HYADAPT = PELOTON_BUILD_DIR + "/src/hyadapt"
YCSB = PELOTON_BUILD_DIR + "/src/ycsb"

OUTPUT_FILE = "outputfile.summary"

PROJECTIVITY_DIR = BASE_DIR + "/results/projectivity/"
SELECTIVITY_DIR = BASE_DIR + "/results/selectivity/"
OPERATOR_DIR = BASE_DIR + "/results/operator/"
YCSB_DIR = BASE_DIR + "/results/ycsb/"
VERTICAL_DIR = BASE_DIR + "/results/vertical/"
SUBSET_DIR = BASE_DIR + "/results/subset/"
ADAPT_DIR = BASE_DIR + "/results/adapt/"
WEIGHT_DIR = BASE_DIR + "/results/weight/"
REORG_DIR = BASE_DIR + "/results/reorg/"

LAYOUTS = ("row", "column", "hybrid")
OPERATORS = ("direct", "aggregate")
REORG_LAYOUTS = ("row", "hybrid")

SCALE_FACTOR = 1000.0

SELECTIVITY = (0.2, 0.4, 0.6, 0.8, 1.0)
PROJECTIVITY = (0.01, 0.1, 0.5, 1.0)
SUBSET_RATIOS = (0.2, 0.4, 0.6, 0.8, 1)
ACCESS_NUM_GROUPS = (1, 2, 4, 8, 16)

SUBSET_SINGLE_GROUP_EXPERIMENT = "1"
SUBSET_MULTIPLE_GROUP_EXPERIMENT = "2"

OP_PROJECTIVITY = (0.01, 0.1, 1.0)
OP_COLUMN_COUNT = 100
OP_SELECTIVITY = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

COLUMN_COUNTS = (50, 200)
WRITE_RATIOS = (0, 0.1)
TUPLES_PER_TILEGROUP = (10, 100, 1000, 10000)
NUM_GROUPS = 5

THETAS = (0, 0.5)

TRANSACTION_COUNT = 3

NUM_ADAPT_TESTS = 12
REPEAT_ADAPT_TEST = 25
QUERY_COUNT = NUM_ADAPT_TESTS * REPEAT_ADAPT_TEST

SAMPLE_WEIGHTS = (0.001, 0.01, 0.1)
NUM_WEIGHT_TEST = 10
REPEAT_WEIGHT_TEST = 20
WEIGHT_QUERY_COUNT = NUM_WEIGHT_TEST * REPEAT_WEIGHT_TEST

REORG_SCALE_FACTORS = (100, 1000)
REORG_QUERY_COUNT = 50

PROJECTIVITY_EXPERIMENT = 1
SELECTIVITY_EXPERIMENT = 2
OPERATOR_EXPERIMENT = 3
VERTICAL_EXPERIMENT= 4
SUBSET_EXPERIMENT= 5
ADAPT_EXPERIMENT = 6
WEIGHT_EXPERIMENT = 7
REORG_EXPERIMENT = 8

YCSB_EXPERIMENT = 1

YCSB_SCALE_FACTOR = 100.0
YCSB_TRANSACTION_COUNT = 100

YCSB_OPERATIONS = ["Read", "Scan", "Insert", "Delete", "Update", "RMW"]

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

def next_power_of_10(n):
    return (10 ** math.ceil(math.log(n, 10)))

def get_upper_bound(n):
    return (math.ceil(n / YAXIS_ROUND) * YAXIS_ROUND)

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

def create_bar_legend():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(9, 0.5))

    num_items = len(LAYOUTS);
    ind = np.arange(1)
    margin = 0.10
    width = ((1.0 - 2 * margin) / num_items) * 2
    data = [1]

    bars = [None] * (len(LAYOUTS) + 1) * 2

    # TITLE
    idx = 0    
    bars[idx] = ax1.bar(ind + margin + ((idx) * width), data, width,
                        color = 'w',
                        linewidth=0)
    
    idx = 0    
    for group in xrange(len(LAYOUTS)):
        bars[idx + 1] = ax1.bar(ind + margin + ((idx + 1) * width), data, width,
                              color=OPT_COLORS[idx],
                              hatch=OPT_PATTERNS[idx * 2],
                              linewidth=BAR_LINEWIDTH)
        
        idx = idx + 1

    TITLE = "Storage Models : "
    LABELS = [TITLE, "NSM", "DSM", "FSM"]

    # LEGEND
    figlegend.legend(bars, LABELS, prop=LEGEND_FP, 
                     loc=1, ncol=4,
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0, 
                     handleheight=1.5, handlelength=4)

    figlegend.savefig('legend_bar.pdf')

def create_vertical_legend():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(11, 0.5))

    num_items = len(LAYOUTS);
    ind = np.arange(1)
    margin = 0.10
    width = ((1.0 - 2 * margin) / num_items) * 2
    data = [1]

    bars = [None] * (len(TUPLES_PER_TILEGROUP) + 1) * 2

    # TITLE
    idx = 0    
    bars[idx] = ax1.bar(ind + margin + ((idx) * width), data, width,
                        color = 'w',
                        linewidth=0)

    idx = 0
    for group in xrange(len(TUPLES_PER_TILEGROUP)):
        bars[idx + 1] = ax1.bar(ind + margin + ((idx + 1) * width), data, width,
                              color=OPT_COLORS[idx],
                              linewidth=BAR_LINEWIDTH)
        
        idx = idx + 1


    TITLE = "Tuples Per Tile Group : "
    LABELS = [TITLE, 10, 100, 1000, 10000]
    
    # LEGEND
    figlegend.legend(bars, LABELS, prop=LEGEND_FP,
                     loc=1, ncol=5, 
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0, 
                     handleheight=1.5, handlelength=4)

    figlegend.savefig('legend_vertical.pdf')

def create_legend():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(9, 0.5))
    idx = 0
    lines = [None] * (len(LAYOUTS) + 1)
    data = [1]
    x_values = [1]

    TITLE = "Storage Models : "
    LABELS = [TITLE, "NSM", "DSM", "FSM"]
        
    lines[idx], = ax1.plot(x_values, data, linewidth = 0)    
    idx = 0
    
    for group in xrange(len(LAYOUTS)):
        lines[idx + 1], = ax1.plot(x_values, data, 
                               color=OPT_LINE_COLORS[idx], linewidth=OPT_LINE_WIDTH,
                               marker=OPT_MARKERS[idx], markersize=OPT_MARKER_SIZE, label=str(group))

        idx = idx + 1

    # LEGEND
    figlegend.legend(lines, LABELS, prop=LEGEND_FP,
                     loc=1, ncol=4, mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0, handlelength=4)
        
    figlegend.savefig('legend.pdf')


def create_projectivity_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    x_values = PROJECTIVITY
    N = len(x_values)
    x_labels = PROJECTIVITY

    layouts = ["NSM", "DSM", "FSM"]

    ind = np.arange(N)
    margin = 0.15
    width = ((1.0 - 2 * margin) / N)
    bars = [None] * len(layouts) * N

    print(datasets)

    for group in xrange(len(datasets)):
        # GROUP
        latencies = []

        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    latencies.append(datasets[group][line][col])

        LOG.info("%s group_data = %s ", layouts, str(latencies))

        bars[group] = ax1.bar(ind + margin + (group * width), latencies, width,
                              color=OPT_COLORS[group],
                              hatch=OPT_PATTERNS[group*2],
                              linewidth=BAR_LINEWIDTH)


    # GRID
    axes = ax1.get_axes()
    #axes.set_ylim(0.01, 1000000)
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    #ax1.set_ylim([YAXIS_MIN, YAXIS_MAX])

    # X-AXIS
    ax1.set_xlabel("Fraction of Attributes Projected", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)
    ax1.set_xticks(ind + 0.5)

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
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    #ax1.set_yscale('log', basey=2)

    # X-AXIS
    XAXIS_MIN = 0.1
    XAXIS_MAX = 1.1
    ax1.set_xlabel("Fraction of Tuples Selected", fontproperties=LABEL_FP)
    ax1.set_xlim([XAXIS_MIN, XAXIS_MAX])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_vertical_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = SELECTIVITY
    N = len(x_values)
    x_labels = x_values

    num_items = len(TUPLES_PER_TILEGROUP);
    ind = np.arange(N)
    idx = 0

    # GROUP
    for group_index, group in enumerate(TUPLES_PER_TILEGROUP):
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
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    #ax1.set_yscale('log', basey=10)

    # X-AXIS
    XAXIS_MIN = 0.1
    XAXIS_MAX = 1.1
    ax1.set_xlabel("Fraction of Tuples Selected", fontproperties=LABEL_FP)
    ax1.set_xlim([XAXIS_MIN, XAXIS_MAX])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_operator_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = OP_SELECTIVITY
    N = len(x_values)
    x_labels = x_values

    num_items = len(LAYOUTS);
    ind = np.arange(N)
    idx = 0

    YLIMIT = 0

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

        YLIMIT = max(YLIMIT, max(group_data))

    # GRID
    axes = ax1.get_axes()
    makeGrid(ax1)

    YLIMIT = next_power_of_10(YLIMIT)

    # X-AXIS
    XAXIS_MIN = 0.05
    XAXIS_MAX = 1.05
    ax1.set_xlabel("Fraction of Tuples Selected", fontproperties=LABEL_FP)
    ax1.set_xlim([XAXIS_MIN, XAXIS_MAX])
    x_values = (0.2, 0.4, 0.6, 0.8, 1.0)
    ax1.set_xticks(x_values)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    #ax1.set_yscale('log', basey=2)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_subset_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    x_values = SELECTIVITY
    N = len(x_values)
    x_labels = SELECTIVITY

    ind = np.arange(N)
    margin = 0.15
    width = ((1.0 - 2 * margin) / N)
    bars = [None] * len(SUBSET_RATIOS) * N

    for group in xrange(len(datasets)):
        # GROUP
        latencies = []

        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    latencies.append(datasets[group][line][col])

        LOG.info("%s latencies = %s ", SUBSET_RATIOS[group], str(latencies))

        bars[group] = ax1.bar(ind + margin + (group * width), latencies, width,
                              color=OPT_COLORS[group],
                              hatch=OPT_PATTERNS[group*2],
                              linewidth=BAR_LINEWIDTH)


    # GRID
    axes = ax1.get_axes()
    #axes.set_ylim(0.01, 1000000)
    makeGrid(ax1)

    # Y-AXIS
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()

    # X-AXIS
    ax1.set_xlabel("Fraction of Tuples Selected", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)
    ax1.set_xticks(ind + 0.5)
    ax1.tick_params(axis='x', which='both', bottom='off', top='off')

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    TITLE = "Subset Ratio"
    LABELS = SUBSET_RATIOS
    
    # LEGEND
    ax1.legend(bars, LABELS, prop=LEGEND_FP, 
               loc='upper left',
               title = TITLE,
               ncol=3, shadow=OPT_LEGEND_SHADOW,
               frameon=False, borderaxespad=0.0, 
               handleheight=0.25, handlelength=0.75)

    ax1.get_legend().get_title().set_fontproperties(LABEL_FP)
    ax1.get_legend().get_title().set_position((-55, 0))

    return (fig)

def create_ycsb_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    x_values = YCSB_OPERATIONS
    N = len(x_values)
    x_labels = YCSB_OPERATIONS

    ind = np.arange(N)
    margin = 0.15
    width = ((1.0 - 2 * margin) / N) * 2
    bars = [None] * len(LAYOUTS) * N

    for group in xrange(len(datasets)):
        # GROUP
        latencies = []

        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    latencies.append(datasets[group][line][col])

        LOG.info("%s latencies = %s ", LAYOUTS[group], str(latencies))

        bars[group] = ax1.bar(ind + margin + (group * width), latencies, width,
                              color=OPT_COLORS[group],
                              hatch=OPT_PATTERNS[group*2],
                              linewidth=BAR_LINEWIDTH)


    # GRID
    axes = ax1.get_axes()
    #axes.set_ylim(0.01, 1000000)
    makeGrid(ax1)

    # Y-AXIS
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()

    # X-AXIS
    #ax1.set_xlabel("Number of transactions", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_labels)
    ax1.set_xticks(ind + 0.5)
    ax1.tick_params(axis='x', which='both', bottom='off', top='off')

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_adapt_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = list(xrange(1, QUERY_COUNT + 1))
    N = len(x_values)
    x_labels = x_values

    num_items = len(LAYOUTS);
    ind = np.arange(N)
    idx = 0

    ADAPT_OPT_LINE_WIDTH = 3.0
    ADAPT_OPT_MARKER_SIZE = 5.0

    # GROUP
    for group_index, group in enumerate(LAYOUTS):
        group_data = []

        # LINE
        for line_index, line in enumerate(x_values):
            group_data.append(datasets[group_index][line_index][1])

        LOG.info("%s group_data = %s ", group, str(group_data))

        ax1.plot(x_values, group_data, color=OPT_LINE_COLORS[idx], linewidth=ADAPT_OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx], markersize=ADAPT_OPT_MARKER_SIZE, label=str(group))

        idx = idx + 1

    # GRID
    axes = ax1.get_axes()
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    ax1.set_yscale('log', basey=10)

    # X-AXIS
    ax1.set_xlabel("Query Sequence", fontproperties=LABEL_FP)
    major_ticks = np.arange(0, QUERY_COUNT + 1, REPEAT_ADAPT_TEST)
    ax1.set_xticks(major_ticks)
    
    #for major_tick in major_ticks[1:-1]:
    #    ax1.axvline(major_tick, color='0.5', linestyle='dashed', linewidth=ADAPT_OPT_LINE_WIDTH)    
    
    # LABELS
    y_mark = 0.9
    x_mark_count = 1.0/NUM_ADAPT_TESTS
    x_mark_offset = x_mark_count/2 - x_mark_count/4
    x_marks = np.arange(0, 1, x_mark_count)
    
    ADAPT_LABELS = (["Select", "Insert", "Select", "Insert",
                     "Select", "Insert", "Select", "Insert",
                     "Select", "Insert", "Select", "Insert"])
    
    for idx, x_mark in enumerate(x_marks):
            ax1.text(x_mark + x_mark_offset, 
                     y_mark, 
                     ADAPT_LABELS[idx], 
                     transform=ax1.transAxes,
                     bbox=dict(facecolor='skyblue', alpha=0.5))
        
    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

def create_weight_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = list(xrange(1, WEIGHT_QUERY_COUNT + 1))
    N = len(x_values)
    x_labels = x_values

    num_items = len(SAMPLE_WEIGHTS);
    ind = np.arange(N)
    idx = 0
    lines = [None] * (len(SAMPLE_WEIGHTS) + 1)

    ADAPT_OPT_LINE_WIDTH = 3.0
    ADAPT_OPT_MARKER_SIZE = 5.0

    # GROUP
    for group_index, group in enumerate(SAMPLE_WEIGHTS):
        group_data = []

        # LINE
        for line_index, line in enumerate(x_values):
            group_data.append(datasets[group_index][line_index][1])

        LOG.info("%s group_data = %s ", group, str(group_data))

        lines[idx], = ax1.plot(x_values, group_data, color=OPT_LINE_COLORS[idx], linewidth=ADAPT_OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx], markersize=ADAPT_OPT_MARKER_SIZE, label=str(group))

        idx = idx + 1

    # GRID
    axes = ax1.get_axes()
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Split Point", fontproperties=LABEL_FP)
    #ax1.set_yscale('log', basey=10)

    # X-AXIS
    ax1.set_xlabel("Query Sequence", fontproperties=LABEL_FP)
    major_ticks = np.arange(0, WEIGHT_QUERY_COUNT + 1, REPEAT_WEIGHT_TEST)
    ax1.set_xticks(major_ticks)
    
    #for major_tick in major_ticks[1:-1]:
    #    ax1.axvline(major_tick, color='0.5', linestyle='dashed', linewidth=ADAPT_OPT_LINE_WIDTH)    

    TITLE = "Weight"
    LABELS = SAMPLE_WEIGHTS
    
    # LEGEND
    ax1.legend(lines, LABELS, prop=LABEL_FP, title = TITLE,
               loc=1, ncol=1, mode="expand", shadow=OPT_LEGEND_SHADOW,
               frameon=False, borderaxespad=0.0, handlelength=2)

    ax1.get_legend().get_title().set_fontproperties(LABEL_FP)
    ax1.get_legend().get_title().set_position((-110, 0))

    return (fig)

def create_reorg_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = list(xrange(1, REORG_QUERY_COUNT + 1))
    N = len(x_values)
    x_labels = x_values

    num_items = len(REORG_LAYOUTS);
    ind = np.arange(N)
    idx = 0
    lines = [None] * (len(REORG_LAYOUTS) + 1)

    ADAPT_OPT_LINE_WIDTH = 3.0
    ADAPT_OPT_MARKER_SIZE = 5.0

    # GROUP
    for group_index, group in enumerate(REORG_LAYOUTS):
        group_data = []

        # LINE
        for line_index, line in enumerate(x_values):
            group_data.append(datasets[group_index][line_index][1])

        LOG.info("%s group_data = %s ", group, str(group_data))

        lines[idx], = ax1.plot(x_values, group_data, color=OPT_LINE_COLORS[idx], linewidth=ADAPT_OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx], markersize=ADAPT_OPT_MARKER_SIZE, label=str(group))

        idx = idx + 1

    # GRID
    axes = ax1.get_axes()
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    ax1.set_yscale('log', basey=10)

    # X-AXIS
    REORG_INTERVAL = 5
    ax1.set_xlabel("Query Sequence", fontproperties=LABEL_FP)
    major_ticks = np.arange(0, REORG_QUERY_COUNT + 1, REORG_INTERVAL)
    ax1.set_xticks(major_ticks)
    
    #for major_tick in major_ticks[1:-1]:
    #    ax1.axvline(major_tick, color='0.5', linestyle='dashed', linewidth=ADAPT_OPT_LINE_WIDTH)    

    TITLE = "Reorganization Type"
    LABELS = ("Immediate", "Incremental")
    
    # LEGEND
    ax1.legend(lines, LABELS, prop=LABEL_FP, title = TITLE,
               loc=0, ncol=1, shadow=OPT_LEGEND_SHADOW,
               frameon=False, borderaxespad=0.0, handlelength=2)

    ax1.get_legend().get_title().set_fontproperties(LABEL_FP)
    ax1.get_legend().get_title().set_position((0, 0))

    return (fig)

###################################################################################
# PLOT HELPERS
###################################################################################

# PROJECTIVITY -- PLOT
def projectivity_plot():

    column_count_type = 0
    for column_count in COLUMN_COUNTS:
        column_count_type = column_count_type + 1

        for write_ratio in WRITE_RATIOS:

            for operator in OPERATORS:
                print(operator)
                datasets = []

                for layout in LAYOUTS:
                    data_file = PROJECTIVITY_DIR + "/" + layout + "/" + operator + "/" + str(column_count) + "/" + str(write_ratio) + "/" + "projectivity.csv"

                    dataset = loadDataFile(4, 2, data_file)
                    datasets.append(dataset)

                fig = create_projectivity_bar_chart(datasets)

                if write_ratio == 0:
                    write_mix = "rd"
                else:
                    write_mix = "rw"

                if column_count_type == 1:
                    table_type = "narrow"
                else:
                    table_type = "wide"

                fileName = "projectivity-" + operator + "-" + table_type + "-" + write_mix + ".pdf"

                saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/2.0)

# SELECTIVITY -- PLOT
def selectivity_plot():

    column_count_type = 0
    for column_count in COLUMN_COUNTS:
        column_count_type = column_count_type + 1

        for write_ratio in WRITE_RATIOS:

            for operator in OPERATORS:
                print(operator)
                datasets = []

                for layout in LAYOUTS:
                    data_file = SELECTIVITY_DIR + "/" + layout + "/" + operator  + "/" + str(column_count) + "/" + str(write_ratio) + "/" + "selectivity.csv"

                    dataset = loadDataFile(10, 2, data_file)
                    datasets.append(dataset)

                fig = create_selectivity_line_chart(datasets)

                if write_ratio == 0:
                    write_mix = "rd"
                else:
                    write_mix = "rw"

                if column_count_type == 1:
                    table_type = "narrow"
                else:
                    table_type = "wide"

                fileName = "selectivity-" + operator + "-" + table_type + "-" + write_mix + ".pdf"

                saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/2.0)


# VERTICAL -- PLOT
def vertical_plot():

    column_count_type = 0
    for column_count in COLUMN_COUNTS:
        column_count_type = column_count_type + 1

        for write_ratio in WRITE_RATIOS:
            datasets = []

            for tuples_per_tg in TUPLES_PER_TILEGROUP:

                data_file = VERTICAL_DIR + "/" + str(tuples_per_tg) + "/" + str(column_count) + "/" + str(write_ratio) + "/" + "vertical.csv"

                dataset = loadDataFile(10, 2, data_file)
                datasets.append(dataset)

            fig = create_vertical_line_chart(datasets)

            if write_ratio == 0:
                write_mix = "rd"
            else:
                write_mix = "rw"

            if column_count_type == 1:
                table_type = "narrow"
            else:
                table_type = "wide"

            fileName = "vertical-" + table_type + "-" + write_mix + ".pdf"

            saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/2.0)


# OPERATOR -- PLOT
def operator_plot():

    column_count = OP_COLUMN_COUNT
    for write_ratio in WRITE_RATIOS:

        projectivity_type = 0
        for projectivity in OP_PROJECTIVITY:
            projectivity_type = projectivity_type + 1
            print(projectivity)
            datasets = []

            for layout in LAYOUTS:
                if projectivity == 1.0: projectivity = 1
                data_file = OPERATOR_DIR + "/" + layout + "/" + str(projectivity) + "/" + str(column_count) + "/" + str(write_ratio) + "/" + "operator.csv"

                dataset = loadDataFile(10, 2, data_file)
                datasets.append(dataset)

            fig = create_operator_line_chart(datasets)

            if write_ratio == 0:
                write_mix = "rd"
            else:
                write_mix = "rw"

            fileName = "operator-" + str(projectivity_type) + "-" + write_mix + ".pdf"

            saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/2.0)


# YCSB -- PLOT
def ycsb_plot():

    column_count = 200
    datasets = []

    for layout in LAYOUTS:
        data_file = YCSB_DIR + "/" + layout + "/" + str(column_count) + "/" + "ycsb.csv"

        dataset = loadDataFile(6, 2, data_file)
        datasets.append(dataset)

    fig = create_ycsb_bar_chart(datasets)

    fileName = "ycsb.pdf"

    saveGraph(fig, fileName, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/2.0)

# SUBSET -- PLOT
def subset_plot():

    datasets = []
    for subset_ratio in SUBSET_RATIOS:
        data_file = SUBSET_DIR + "/" + SUBSET_SINGLE_GROUP_EXPERIMENT + "/" + str(subset_ratio) + "/" + "subset.csv"

        dataset = loadDataFile(5, 2, data_file)
        datasets.append(dataset)

    fig = create_subset_bar_chart(datasets)

    fileName = "subset-single.pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/2.0)

    datasets = []
    for access_num_group in ACCESS_NUM_GROUPS:
        data_file = SUBSET_DIR + "/" + SUBSET_MULTIPLE_GROUP_EXPERIMENT + "/" + str(access_num_group) + "/" + "subset.csv"

        dataset = loadDataFile(5, 2, data_file)
        datasets.append(dataset)

    fig = create_subset_bar_chart(datasets)

    fileName = "subset-multiple.pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/2.0)

# ADAPT -- PLOT
def adapt_plot():

    ADAPT_COLUMN_COUNT = COLUMN_COUNTS[1]
    datasets = []

    for layout in LAYOUTS:
        data_file = ADAPT_DIR + "/" + str(ADAPT_COLUMN_COUNT) + "/" + layout + "/" + "adapt.csv"

        dataset = loadDataFile(QUERY_COUNT, 2, data_file)                    
        datasets.append(dataset)

    fig = create_adapt_line_chart(datasets)

    fileName = "adapt.pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH * 3, height=OPT_GRAPH_HEIGHT/1.5)

# WEIGHT -- PLOT
def weight_plot():

    datasets = []
    for sample_weight in SAMPLE_WEIGHTS:
        data_file = WEIGHT_DIR + "/" + str(sample_weight) + "/" + "weight.csv"

        dataset = loadDataFile(WEIGHT_QUERY_COUNT, 2, data_file)                    
        datasets.append(dataset)

    fig = create_weight_line_chart(datasets)

    fileName = "weight.pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/2.0)

# REORG -- PLOT
def reorg_plot():

    for reorg_scale_factor in REORG_SCALE_FACTORS:
        datasets = []
        
        for layout in REORG_LAYOUTS:
            data_file = REORG_DIR + "/" + str(reorg_scale_factor) + "/" + str(layout) + "/" + "reorg.csv"
    
            dataset = loadDataFile(REORG_QUERY_COUNT, 2, data_file)                    
            datasets.append(dataset)

        fig = create_reorg_line_chart(datasets)
    
        fileName = "reorg_" + str(reorg_scale_factor) + ".pdf"

        saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT/2.0)

###################################################################################
# EVAL HELPERS
###################################################################################

# CLEAN UP RESULT DIR
def clean_up_dir(result_directory):

    subprocess.call(['rm', '-rf', result_directory])
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

# RUN EXPERIMENT
def run_experiment(program,
                   scale_factor,
                   transaction_count,
                   experiment_type):

    # cleanup
    subprocess.call(["rm -f " + OUTPUT_FILE], shell=True)

    subprocess.call([program,
                     "-e", str(experiment_type),
                     "-k", str(scale_factor),
                     "-t", str(transaction_count)])


# COLLECT STATS
def collect_stats(result_dir,
                  result_file_name,
                  category):

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
        column_count = data[4]
        write_ratio = data[5]
        subset_experiment_type = data[6]
        access_num_group = data[7]
        subset_ratio = data[8]
        tuples_per_tg = data[9]
        txn_itr = data[10]
        theta = data[11]
        split_point = data[12]
        sample_weight = data[13]
        scale_factor = data[14]
        stat = data[15]

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

        # MAKE RESULTS FILE DIR
        if category == PROJECTIVITY_EXPERIMENT or category == SELECTIVITY_EXPERIMENT:
            result_directory = result_dir + "/" + layout + "/" + operator + "/" + column_count + "/" + write_ratio
        elif category == OPERATOR_EXPERIMENT:
            result_directory = result_dir + "/" + layout + "/" + str(projectivity) + "/" + column_count + "/" + write_ratio
        elif category == VERTICAL_EXPERIMENT:
            result_directory = result_dir + "/" + str(tuples_per_tg) + "/" + column_count + "/" + write_ratio
        elif category == SUBSET_EXPERIMENT:
            if subset_experiment_type == SUBSET_SINGLE_GROUP_EXPERIMENT:
                result_directory = result_dir + "/" + str(subset_experiment_type) + "/" + str(subset_ratio)
            elif subset_experiment_type == SUBSET_MULTIPLE_GROUP_EXPERIMENT:
                result_directory = result_dir + "/" + str(subset_experiment_type) + "/" + str(access_num_group)
        elif category == ADAPT_EXPERIMENT:
            result_directory = result_dir + "/" + column_count + "/" + layout
        elif category == WEIGHT_EXPERIMENT:
            result_directory = result_dir + "/" + str(sample_weight)
        elif category == REORG_EXPERIMENT:
            result_directory = result_dir + "/" + str(scale_factor) + "/" + layout

        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        file_name = result_directory + "/" + result_file_name

        result_file = open(file_name, "a")

        # WRITE OUT STATS
        if category == PROJECTIVITY_EXPERIMENT:
            result_file.write(str(projectivity) + " , " + str(stat) + "\n")
        elif category == SELECTIVITY_EXPERIMENT or category == OPERATOR_EXPERIMENT or category == VERTICAL_EXPERIMENT or category == SUBSET_EXPERIMENT:
            result_file.write(str(selectivity) + " , " + str(stat) + "\n")
        elif category == ADAPT_EXPERIMENT or category == REORG_EXPERIMENT:
            result_file.write(str(txn_itr) + " , " + str(stat) + "\n")
        elif category == WEIGHT_EXPERIMENT:
            result_file.write(str(txn_itr) + " , " + str(split_point) + "\n")

        result_file.close()

# COLLECT STATS
def collect_ycsb_stats(result_dir,
                       result_file_name):

    fp = open(OUTPUT_FILE)
    lines = fp.readlines()
    fp.close()

    for line in lines:
        data = line.split()

        # Collect info
        layout = data[0]
        operator = data[1]
        column_count = data[2]
        stat = data[3]

        if(layout == "0"):
            layout = "row"
        elif(layout == "1"):
            layout = "column"
        elif(layout == "2"):
            layout = "hybrid"

        result_directory = result_dir + "/" + layout + "/" + column_count

        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        file_name = result_directory + "/" + result_file_name

        result_file = open(file_name, "a")
        result_file.write(str(operator) + " , " + str(stat) + "\n")
        result_file.close()

###################################################################################
# EVAL
###################################################################################

# PROJECTIVITY -- EVAL
def projectivity_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(PROJECTIVITY_DIR)

    # RUN EXPERIMENT
    run_experiment(HYADAPT, SCALE_FACTOR,
                   TRANSACTION_COUNT, PROJECTIVITY_EXPERIMENT)

    # COLLECT STATS
    collect_stats(PROJECTIVITY_DIR, "projectivity.csv", PROJECTIVITY_EXPERIMENT)

# SELECTIVITY -- EVAL
def selectivity_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(SELECTIVITY_DIR)

    # RUN EXPERIMENT
    run_experiment(HYADAPT, SCALE_FACTOR,
                   TRANSACTION_COUNT, SELECTIVITY_EXPERIMENT)

    # COLLECT STATS
    collect_stats(SELECTIVITY_DIR, "selectivity.csv", SELECTIVITY_EXPERIMENT)

# OPERATOR -- EVAL
def operator_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(OPERATOR_DIR)

    # RUN EXPERIMENT
    run_experiment(HYADAPT, SCALE_FACTOR,
                   TRANSACTION_COUNT, OPERATOR_EXPERIMENT)

    # COLLECT STATS
    collect_stats(OPERATOR_DIR, "operator.csv", OPERATOR_EXPERIMENT)

# VERTICAL -- EVAL
def vertical_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(VERTICAL_DIR)

    # RUN EXPERIMENT
    run_experiment(HYADAPT, SCALE_FACTOR,
                   TRANSACTION_COUNT, VERTICAL_EXPERIMENT)

    # COLLECT STATS
    collect_stats(VERTICAL_DIR, "vertical.csv", VERTICAL_EXPERIMENT)

# YCSB -- EVAL
def ycsb_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(YCSB_DIR)

    # RUN EXPERIMENT
    run_experiment(YCSB, YCSB_SCALE_FACTOR,
                   YCSB_TRANSACTION_COUNT, YCSB_EXPERIMENT)

    # COLLECT STATS
    collect_ycsb_stats(YCSB_DIR, "ycsb.csv")

# SUBSET -- EVAL
def subset_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(SUBSET_DIR)

    # RUN EXPERIMENT
    run_experiment(HYADAPT, SCALE_FACTOR,
                   TRANSACTION_COUNT, SUBSET_EXPERIMENT)

    # COLLECT STATS
    collect_stats(SUBSET_DIR, "subset.csv", SUBSET_EXPERIMENT)

# ADAPT -- EVAL
def adapt_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(ADAPT_DIR)

    # RUN EXPERIMENT
    run_experiment(HYADAPT, SCALE_FACTOR,
                   TRANSACTION_COUNT, ADAPT_EXPERIMENT)

    # COLLECT STATS
    collect_stats(ADAPT_DIR, "adapt.csv", ADAPT_EXPERIMENT)
    
# WEIGHT -- EVAL
def weight_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(WEIGHT_DIR)

    # RUN EXPERIMENT
    run_experiment(HYADAPT, SCALE_FACTOR,
                   TRANSACTION_COUNT, WEIGHT_EXPERIMENT)

    # COLLECT STATS
    collect_stats(WEIGHT_DIR, "weight.csv", WEIGHT_EXPERIMENT)        

# REORG -- EVAL
def reorg_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(REORG_DIR)

    # RUN EXPERIMENT
    run_experiment(HYADAPT, SCALE_FACTOR,
                   TRANSACTION_COUNT, REORG_EXPERIMENT)

    # COLLECT STATS
    collect_stats(REORG_DIR, "reorg.csv", REORG_EXPERIMENT)    

###################################################################################
# MAIN
###################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Tilegroup Experiments')

    parser.add_argument("-p", "--projectivity", help='eval projectivity', action='store_true')
    parser.add_argument("-s", "--selectivity", help='eval selectivity', action='store_true')
    parser.add_argument("-o", "--operator", help='eval operator', action='store_true')
    parser.add_argument("-y", "--ycsb", help='eval ycsb', action='store_true')
    parser.add_argument("-v", "--vertical", help='eval vertical', action='store_true')
    parser.add_argument("-u", "--subset", help='eval subset', action='store_true')
    parser.add_argument("-z", "--adapt", help='eval adapt', action='store_true')
    parser.add_argument("-w", "--weight", help='eval weight', action='store_true')
    parser.add_argument("-r", "--reorg", help='eval reorg', action='store_true')

    parser.add_argument("-a", "--projectivity_plot", help='plot projectivity', action='store_true')
    parser.add_argument("-b", "--selectivity_plot", help='plot selectivity', action='store_true')
    parser.add_argument("-c", "--operator_plot", help='plot operator', action='store_true')
    parser.add_argument("-d", "--ycsb_plot", help='plot ycsb', action='store_true')
    parser.add_argument("-e", "--vertical_plot", help='plot vertical', action='store_true')
    parser.add_argument("-f", "--subset_plot", help='plot subset', action='store_true')
    parser.add_argument("-g", "--adapt_plot", help='plot adapt', action='store_true')
    parser.add_argument("-i", "--weight_plot", help='plot weight', action='store_true')
    parser.add_argument("-j", "--reorg_plot", help='plot reorg', action='store_true')

    args = parser.parse_args()

    if args.projectivity:
        projectivity_eval()

    if args.projectivity_plot:
        projectivity_plot();

    if args.selectivity:
        selectivity_eval()

    if args.selectivity_plot:
        selectivity_plot();

    if args.operator:
        operator_eval()

    if args.operator_plot:
        operator_plot();

    if args.ycsb:
        ycsb_eval()

    if args.ycsb_plot:
        ycsb_plot()

    if args.vertical:
        vertical_eval()

    if args.vertical_plot:
        vertical_plot()

    if args.subset:
        subset_eval()

    if args.subset_plot:
        subset_plot()

    if args.adapt:
        adapt_eval()

    if args.adapt_plot:
        adapt_plot()

    if args.weight:
        weight_eval()

    if args.weight_plot:
        weight_plot()

    if args.reorg:
        reorg_eval()

    if args.reorg_plot:
        reorg_plot()

    #create_legend()
    #create_bar_legend()
    #create_vertical_legend()


