# Copyright (c) 2012-2021 Dynatrace LLC. All rights reserved.
#
# This software and associated documentation files (the "Software")
# are being made available by Dynatrace LLC for purposes of
# illustrating the implementation of certain algorithms which have
# been published by Dynatrace LLC. Permission is hereby granted,
# free of charge, to any person obtaining a copy of the Software,
# to view and use the Software for internal, non-productive,
# non-commercial purposes only – the Software may not be used to
# process live data or distributed, sublicensed, modified and/or
# sold either alone or as part of or in combination with any other
# software.
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from collections import OrderedDict
import csv
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import math
import glob
import matplotlib.ticker as mtick
import color_defs

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\RequirePackage[T1]{fontenc} \RequirePackage[tt=false, type1=true]{libertine} \RequirePackage[varqu]{zi4} \RequirePackage[libertine]{newtxmath}')


expected_linestyle="dotted"
expected_color=color_defs.colorGray
expected_linewidth=2

def read_data(data_file):

    info = {}

    with open(data_file, 'r') as file:
        reader = csv.reader(file, skipinitialspace=True, delimiter=';')
        row_counter = 0
        headers = []
        values = []
        for r in reader:
            if row_counter == 0:
                for i in r:
                    if i != "":
                        g = i.split("=")
                        info[g[0]] = g[1]

            elif row_counter == 1:
                for i in r:
                    if i != "":
                        headers.append(i)
                        values.append([])
            elif row_counter >= 2:
                k = 0
                for i in r:
                    if i != "":
                        values[k].append(float(i))
                        k += 1
            row_counter += 1

    data = {h:v for h,v in zip(headers, values)}
    size = row_counter - 2
    return info, data, size

def calc_relative_error(true_cardinalities, mean):
    assert(len(true_cardinalities) == len(mean))
    return [(m - t)/t for t, m in zip(true_cardinalities, mean)]

def calculate_expected_relative_mse(m,b):
    return math.sqrt(((b + 1)/(b-1)*math.log(b)-1)/m)

filenames = glob.glob("data/cardinality_test*.csv")

data = []
for filename in filenames:
    data.append(read_data(filename))


def get_data(m,b):
    card = None
    for d in data:
        if d[0]["name"] == "MinHash" or d[0]["name"] == "HyperMinHash" : continue
        if int(d[0]["numRegisters"]) == m and float(d[0]["base"]) == b:
            if d[0]["name"] == "GeneralizedHyperLogLog":
                data_ghll = d[1]
            elif d[0]["name"] == "SetSketch1":
                data_ss1 = d[1]
            elif d[0]["name"] == "SetSketch2":
                data_ss2 = d[1]
            else:
                assert(False)

            if card is None:
                card = d[1]["true cardinality"]
            else:
                assert(card == d[1]["true cardinality"])
    assert(data_ghll is not None)
    assert(data_ss1 is not None)
    assert(data_ss2 is not None)

    return card, data_ss1, data_ss2, data_ghll

def format_x_axis(ax):
    ax.set_xscale("log", basex=10)
    ax.set_xlim([1,1e7])

def plot_bias_chart(ax, m, b, prefix):
    card, data_ss1, data_ss2, data_ghll = get_data(m,b)

    data_ss1_mean = data_ss1[prefix + " mean"][1:]
    data_ss2_mean = data_ss2[prefix + " mean"][1:]
    data_ghll_mean = data_ghll[prefix + " mean"][1:]

    ax.plot([1,1e8], [0,0], linewidth=expected_linewidth, linestyle=expected_linestyle, color = expected_color)
    ax.plot(card[1:], [100*x for x in calc_relative_error(card[1:], data_ss1_mean)], linewidth=1, linestyle="solid", label="SetSketch1", color = color_defs.colorSetSketch1)
    ax.plot(card[1:], [100*x for x in calc_relative_error(card[1:], data_ss2_mean)], linewidth=1, linestyle="solid", label="SetSketch2", color = color_defs.colorSetSketch2)
    ax.plot(card[1:], [100*x for x in calc_relative_error(card[1:], data_ghll_mean)], linewidth=1, linestyle="solid", label="GHLL", color = color_defs.colorGHLL)
    format_x_axis(ax)

def plot_rmse_chart(ax, m, b, prefix):
    card, data_ss1, data_ss2, data_ghll = get_data(m,b)

    data_ss1_mse = data_ss1[prefix + " mse"]
    data_ss2_mse= data_ss2[prefix + " mse"]
    data_ghll_mse = data_ghll[prefix + " mse"]

    expected = 100*calculate_expected_relative_mse(m,b)
    ax.plot([1,1e8], [expected,expected], linewidth=expected_linewidth, linestyle=expected_linestyle, color = expected_color)
    ax.plot(card[1:], [100*math.sqrt(data_ss1_mse[i]) / card[i] for i in range(1, len(card))], linewidth=1, linestyle="solid", label="SetSketch1", color = color_defs.colorSetSketch1)
    ax.plot(card[1:], [100*math.sqrt(data_ss2_mse[i]) / card[i] for i in range(1, len(card))], linewidth=1, linestyle="solid", label="SetSketch2", color = color_defs.colorSetSketch2)
    ax.plot(card[1:], [100*math.sqrt(data_ghll_mse[i]) / card[i] for i in range(1, len(card))], linewidth=1, linestyle="solid", label="GHLL", color = color_defs.colorGHLL)
    format_x_axis(ax)

def plot_kurtosis_chart(ax, m, b, prefix):
    card, data_ss1, data_ss2, data_ghll = get_data(m,b)

    data_ss1_kurtosis = data_ss1[prefix + " kurtosis"][1:]
    data_ss2_kurtosis = data_ss2[prefix + " kurtosis"][1:]
    data_ghll_kurtosis = data_ghll[prefix + " kurtosis"][1:]

    ax.plot([1,1e8], [3,3], linewidth=expected_linewidth, linestyle=expected_linestyle, color = expected_color)
    ax.plot(card[1:], data_ss1_kurtosis, linewidth=1, linestyle="solid", label="SetSketch1", color = color_defs.colorSetSketch1)
    ax.plot(card[1:], data_ss2_kurtosis, linewidth=1, linestyle="solid", label="SetSketch2", color = color_defs.colorSetSketch2)
    ax.plot(card[1:], data_ghll_kurtosis, linewidth=1, linestyle="solid", label="GHLL", color = color_defs.colorGHLL)
    format_x_axis(ax)
    ax.set_yscale("log", basey=10)
    ax.set_yticks([1,1e1,1e2,1e3,1e4,1e5])

def make_chart(prefix):
    fig = plt.figure(figsize=(6, 6), constrained_layout=False)

    outer_grid = fig.add_gridspec(ncols=1, nrows=2, wspace=0., hspace=0.1)
    inner_grids = [outer_grid[i, 0].subgridspec(ncols=2, nrows=3, wspace=0.04, hspace=0.12) for i in range(0,2)]

    ax_m256_b2_bias = fig.add_subplot(inner_grids[0][0, 0])
    ax_m256_b2_rmse = fig.add_subplot(inner_grids[0][1, 0])
    ax_m256_b2_kurtosis = fig.add_subplot(inner_grids[0][2, 0])
    ax_m256_b1_001_bias = fig.add_subplot(inner_grids[0][0, 1])
    ax_m256_b1_001_rmse = fig.add_subplot(inner_grids[0][1, 1])
    ax_m256_b1_001_kurtosis = fig.add_subplot(inner_grids[0][2, 1])
    ax_m4096_b2_bias = fig.add_subplot(inner_grids[1][0, 0])
    ax_m4096_b2_rmse = fig.add_subplot(inner_grids[1][1, 0])
    ax_m4096_b2_kurtosis = fig.add_subplot(inner_grids[1][2, 0])
    ax_m4096_b1_001_bias = fig.add_subplot(inner_grids[1][0, 1])
    ax_m4096_b1_001_rmse = fig.add_subplot(inner_grids[1][1, 1])
    ax_m4096_b1_001_kurtosis = fig.add_subplot(inner_grids[1][2, 1])

    plot_bias_chart(ax_m256_b2_bias, 256, 2, prefix)
    plot_bias_chart(ax_m256_b1_001_bias, 256, 1.001, prefix)
    plot_rmse_chart(ax_m256_b2_rmse, 256, 2, prefix)
    plot_rmse_chart(ax_m256_b1_001_rmse, 256, 1.001, prefix)
    plot_kurtosis_chart(ax_m256_b2_kurtosis, 256, 2, prefix)
    plot_kurtosis_chart(ax_m256_b1_001_kurtosis, 256, 1.001, prefix)

    plot_bias_chart(ax_m4096_b2_bias, 4096, 2, prefix)
    plot_bias_chart(ax_m4096_b1_001_bias, 4096, 1.001, prefix)
    plot_rmse_chart(ax_m4096_b2_rmse, 4096, 2, prefix)
    plot_rmse_chart(ax_m4096_b1_001_rmse, 4096, 1.001, prefix)
    plot_kurtosis_chart(ax_m4096_b2_kurtosis, 4096, 2, prefix)
    plot_kurtosis_chart(ax_m4096_b1_001_kurtosis, 4096, 1.001, prefix)

    # y-labels
    for ax,label in [(ax_m256_b1_001_rmse, "$m=256$"), (ax_m4096_b1_001_rmse, "$m=4096$")]:
        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.set_ylabel(label, rotation=270, labelpad=14)

    for ax,label in [(ax_m256_b2_bias, "$b=2$"), (ax_m256_b1_001_bias, "$b=1.001$")]:
        ax2 = ax.twiny()
        ax2.set_xticks([])
        ax2.set_xlabel(label)

    for ax in [ax_m4096_b2_kurtosis, ax_m4096_b1_001_kurtosis]:
        ax.set_xlabel("cardinality")

    for ax in [ax_m256_b2_kurtosis, ax_m4096_b2_kurtosis]:
        ax.set_ylabel("kurtosis")

    for ax in [ax_m256_b2_bias, ax_m4096_b2_bias]:
        ax.set_ylabel("rel. bias (\%)")

    # y-axis limits
    for ax in [ax_m256_b2_bias, ax_m256_b1_001_bias]:
        ax.set_ylim([-0.2, 0.7])

    for ax in [ax_m4096_b2_bias, ax_m4096_b1_001_bias]:
        ax.set_ylim([-0.2/4, 0.7/4])

    for ax in [ax_m256_b2_rmse, ax_m4096_b2_rmse]:
        ax.set_ylabel("rel. RMSE (\%)")

    for ax in [ax_m4096_b2_rmse, ax_m4096_b1_001_rmse]:
        ax.set_ylim([0., 2])

    for ax in [ax_m256_b2_rmse, ax_m256_b1_001_rmse]:
        ax.set_ylim([0., 2*4])

    for ax in [ax_m4096_b2_kurtosis, ax_m4096_b1_001_kurtosis]:
        ax.set_ylim([1, 5e4])

    for ax in [ax_m256_b2_kurtosis, ax_m256_b1_001_kurtosis]:
        ax.set_ylim([1., 5e2])

    # position y-axis labels
    for ax in [ax_m256_b2_rmse, ax_m256_b2_bias, ax_m256_b2_kurtosis, ax_m4096_b2_rmse, ax_m4096_b2_bias, ax_m4096_b2_kurtosis]:
        ax.yaxis.set_label_coords(x=-0.13, y=0.5,transform=ax.transAxes)

    # remove x-axis labels for inner charts
    for ax in [ax_m256_b2_rmse, ax_m256_b2_bias, ax_m4096_b2_rmse, ax_m4096_b2_bias, ax_m256_b1_001_rmse, ax_m256_b1_001_bias, ax_m4096_b1_001_rmse, ax_m4096_b1_001_bias]:
        ax.xaxis.set_ticklabels([])

    # remove y-axis labels for inner charts
    for ax in [ax_m256_b1_001_rmse, ax_m4096_b1_001_rmse, ax_m256_b1_001_bias, ax_m4096_b1_001_bias, ax_m256_b1_001_kurtosis, ax_m4096_b1_001_kurtosis]:
        ax.yaxis.set_ticklabels([])


    # legend
    leg_lines=[]
    leg_labels=[]
    leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_defs.colorSetSketch1))
    leg_labels.append("SetSketch1")
    leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_defs.colorSetSketch2))
    leg_labels.append("SetSketch2")
    leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_defs.colorGHLL))
    leg_labels.append("GHLL")
    leg_lines.append(matplotlib.lines.Line2D([0], [0], linewidth=expected_linewidth, linestyle=expected_linestyle, color = expected_color))
    leg_labels.append("expected")
    fig.legend(leg_lines, leg_labels, loc="lower center", ncol=len(leg_labels), bbox_to_anchor=(0.5,-0.009))

    fig.subplots_adjust(right=0.96, left=0.082, top=0.97,bottom=0.12)

    fig.savefig("paper/cardinality_" + prefix + ".pdf", format='pdf', dpi=1200, metadata={'creationDate': None})
    plt.close(fig)

make_chart("ml")
make_chart("simple")