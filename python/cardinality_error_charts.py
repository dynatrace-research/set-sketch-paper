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

def calc_upper_error_helper(t,m,e):
    a = e - pow(m - t,2)
    if a >= 0:
        return (m - t + math.sqrt(a))/t
    else:
        return float('inf')

def calc_upper_error(true_cardinalities, mean, mse):
    assert(len(true_cardinalities) == len(mean))
    return [calc_upper_error_helper(t,m,e) for t,m,e in zip(true_cardinalities, mean, mse)]

def calc_lower_error_helper(t,m,e):
    a = e - pow(m - t,2)
    if a >= 0:
        return (m - t - math.sqrt(a))/t
    else:
        return -float('inf')

def calc_lower_error(true_cardinalities, mean, mse):
    assert(len(true_cardinalities) == len(mean))
    return [calc_lower_error_helper(t,m,e) for t,m,e in zip(true_cardinalities, mean, mse)]

def calculate_expected_relative_mse(m,b):
    return math.sqrt(((b + 1)/(b-1)*math.log(b)-1)/m)

def make_chart(ax, data, num_registers, base):
    card = None
    for d in data:
        if int(d[0]["numRegisters"]) == num_registers and float(d[0]["base"]) == base:
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

    expected_relative_mse = calculate_expected_relative_mse(num_registers, base)
    ax.set_xscale("log", basex=10)
    ax.set_xlim(1, card[-1])
    ax.set_ylim([-1.4/math.sqrt(num_registers), 1.4/math.sqrt(num_registers)])
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0,numticks=10))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.fill_between([1, card[-1]],[expected_relative_mse, expected_relative_mse], [-expected_relative_mse, -expected_relative_mse], color=color_defs.colorBackgroundGray, linewidth=1, label="expected RMSE")

    data_ss1_mean = data_ss1["simple mean"][1:]
    data_ss2_mean = data_ss2["simple mean"][1:]
    data_ghll_mean = data_ghll["simple corrected mean"][1:]

    data_ss1_mse = data_ss1["simple mse"][1:]
    data_ss2_mse = data_ss2["simple mse"][1:]
    data_ghll_mse = data_ghll["simple corrected mse"][1:]

    ax.plot(card[1:], calc_relative_error(card[1:], data_ss1_mean), linewidth=1, linestyle="solid", label="SetSketch1", color = color_defs.colorSetSketch1)
    ax.plot(card[1:], calc_lower_error(card[1:], data_ss1_mean, data_ss1_mse), linewidth=1, linestyle="dashed", color = color_defs.colorSetSketch1)
    ax.plot(card[1:], calc_upper_error(card[1:], data_ss1_mean, data_ss1_mse), linewidth=1, linestyle="dashed", color = color_defs.colorSetSketch1)

    ax.plot(card[1:], calc_relative_error(card[1:], data_ss2_mean), linewidth=1, linestyle="solid", label="SetSketch2", color = color_defs.colorSetSketch2)
    ax.plot(card[1:], calc_lower_error(card[1:], data_ss2_mean, data_ss2_mse), linewidth=1, linestyle="dashed", color = color_defs.colorSetSketch2)
    ax.plot(card[1:], calc_upper_error(card[1:], data_ss2_mean, data_ss2_mse), linewidth=1, linestyle="dashed", color = color_defs.colorSetSketch2)

    ax.plot(card[1:], calc_relative_error(card[1:], data_ghll_mean), linewidth=1, linestyle="solid", label="GHLL", color = color_defs.colorGHLL)
    ax.plot(card[1:], calc_lower_error(card[1:], data_ghll_mean, data_ghll_mse), linewidth=1, linestyle="dashed", color = color_defs.colorGHLL)
    ax.plot(card[1:], calc_upper_error(card[1:], data_ghll_mean, data_ghll_mse), linewidth=1, linestyle="dashed", color = color_defs.colorGHLL)

filenames = glob.glob("data/cardinality_test*.csv")

data = []
for filename in filenames:
    data.append(read_data(filename))

fig, ax = plt.subplots(4, 2, sharex=True, sharey="row")
fig.set_size_inches(6, 6)


num_registers = [256, 1024, 4096, 16384]
bases = [2, 1.001]


for i in range(0,2):
    ax[3][i].set_xlabel("cardinality")
    ax2 = ax[0][i].twiny()
    ax2.set_xticks([])
    ax2.set_xlabel("$b=" + '{0:g}'.format(bases[i]) + "$")
for i in range(0,4):
    ax[i][0].set_ylabel("relative error")
    ax[i][0].yaxis.set_major_formatter(mtick.PercentFormatter(1,0))
    if num_registers[i] == 256:
        ax[i][0].set_yticks([-0.05,0,0.05])
        ax[i][0].yaxis.set_minor_locator(matplotlib.ticker.FixedLocator([-0.2 + k*0.01 for k in range(0, 41)]))
    elif num_registers[i] == 1024:
        ax[i][0].set_yticks([-0.04,-0.02,0.0,0.02,0.04])
        ax[i][0].yaxis.set_minor_locator(matplotlib.ticker.FixedLocator([-0.1 + k*0.01 for k in range(0, 21)]))
    elif num_registers[i] == 4096:
        ax[i][0].set_yticks([-0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05])
        ax[i][0].yaxis.set_minor_locator(matplotlib.ticker.FixedLocator([-0.1 + k*0.005 for k in range(0, 41)]))
    else:
        ax[i][0].set_yticks([-0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05])
        ax[i][0].yaxis.set_minor_locator(matplotlib.ticker.FixedLocator([-0.02 + k*0.002 for k in range(0, 21)]))
    ax2 = ax[i][1].twinx()
    ax2.set_yticks([])
    ax2.set_ylabel("$m=" + str(num_registers[i]) + "$", labelpad=12, rotation=270)


for i in range(0, len(num_registers)):
    for j in range(0, len(bases)):
        make_chart(ax[i][j], data, num_registers=num_registers[i], base=bases[j])

 # legend
leg_lines=[]
leg_labels=[]
leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_defs.colorSetSketch1))
leg_labels.append("SetSketch1")
leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_defs.colorSetSketch2))
leg_labels.append("SetSketch2")
leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_defs.colorGHLL))
leg_labels.append("GHLL")
leg_lines.append(matplotlib.patches.Patch(facecolor=color_defs.colorBackgroundGray))
leg_labels.append("expected RMSE")
fig.legend(leg_lines, leg_labels, loc="lower center", ncol=len(leg_labels), bbox_to_anchor=(0.5,-0.009))

fig.subplots_adjust(right=0.97, left=0.082, top=0.97,bottom=0.12, hspace=0.1, wspace=0.05)

fig.savefig('paper/cardinality.pdf', format='pdf', dpi=1200, metadata={'creationDate': None})
plt.close(fig)
