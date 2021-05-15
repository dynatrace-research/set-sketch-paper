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

# from collections import OrderedDict
# import numpy
import csv
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import glob
import matplotlib.ticker as mtick
import color_defs
from collections import namedtuple

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\RequirePackage[T1]{fontenc} \RequirePackage[tt=false, type1=true]{libertine} \RequirePackage[varqu]{zi4} \RequirePackage[libertine]{newtxmath}')

Config = namedtuple("config", ["sketch_name", "aggregation_mode", "register_state_type", "color", "linestyle"])

setsketch1_config = Config("SetSketch1", "stream", "registers with lower bound", color_defs.colorsSketches["SetSketch1"], "solid")
setsketch2_config = Config("SetSketch2", "stream", "registers with lower bound", color_defs.colorsSketches["SetSketch2"], "solid")
ghll_config = Config("GeneralizedHyperLogLog", "stream", "registers", color_defs.colorsSketches["GeneralizedHyperLogLog"], "solid")
ghll_optimized_config = Config("GeneralizedHyperLogLog", "stream", "registers with lower bound", color_defs.colorsSketches["GeneralizedHyperLogLog"], "dotted")
hll_config = Config("HyperLogLog", "stream", "registers", color_defs.colorsSketches["HyperLogLog"], "solid")
hll_optimized_config = Config("HyperLogLog", "stream", "registers with lower bound", color_defs.colorsSketches["HyperLogLog"], "dotted")
minhash_config = Config("MinHash", "stream", "", color_defs.colorsSketches["MinHash"], "dashdot")

# sketch_names = ["SetSketch1", "SetSketch2", "GeneralizedHyperLogLog", "HyperLogLog", "MinHash"]
# sketch_labels = {"SetSketch1":"SetSketch1", "SetSketch2":"SetSketch2", "GeneralizedHyperLogLog":"GHLL", "HyperLogLog":"HLL", "MinHash":"MinHash"}
# linestyles = {"SetSketch1" : "solid","SetSketch2" : "dashed","GeneralizedHyperLogLog" :"dashdot","HyperLogLog" : (0,(1,1)), "MinHash": "dotted"}


def readData(dataFile):

    info = {}

    with open(dataFile, 'r') as file:
        reader = csv.reader(file, skipinitialspace=True, delimiter=';')
        rowCounter = 0
        headers = []
        values = []
        for r in reader:
            if rowCounter == 0:
                for i in r:
                    if i != "":
                        g = i.split("=")
                        assert(len(g) == 1 or len(g)==2)
                        if len(g)==1:
                            info[g[0]] = None
                        else:
                            info[g[0]] = g[1]

            elif rowCounter == 1:
                for i in r:
                    if i != "":
                        headers.append(i)
                        values.append([])
            elif rowCounter >= 2:
                k = 0
                for i in r:
                    if i != "":
                        values[k].append(float(i))
                        k += 1
            rowCounter += 1

    data = {h:v for h,v in zip(headers, values)}
    size = rowCounter - 2
    return info, data, size


def make_dummy_plot(ax, data):
    for d in data:
        if not "dummy" in d[0]:
            continue
        dd = d[1]
        ax.plot(dd["cardinality"], dd["avg time in seconds"])


def make_plot(ax, data, num_registers, base, config):
    for d in data:
        if not "numRegisters" in d[0] or int(d[0]["numRegisters"]) != num_registers:
            continue
        if not "name" in d[0] or d[0]["name"] !=  config.sketch_name:
            continue
        if not "aggregationMode" in d[0] or d[0]["aggregationMode"] !=  config.aggregation_mode:
            continue

        if  config.sketch_name != "MinHash":
            if not "base" in d[0] or float(d[0]["base"]) != base:
                continue
            if not "registerStateType" in d[0] or d[0]["registerStateType"] !=  config.register_state_type:
                continue

        dd = d[1]
        ax.plot(dd["cardinality"], [a/b for a,b in zip(dd["avg time in seconds (excl. allocation)"], dd["cardinality"])], linestyle=config.linestyle, color=config.color)


filenames = glob.glob("data/performance_test*.csv")

data = []
for filename in filenames:
    data.append(readData(filename))

fig, axs = plt.subplots(2, 2,sharey="row", sharex=True)
fig.set_size_inches(6,3.3)

for ax in fig.axes:
    ax.set_xscale("log", basex=10)
    ax.set_yscale("log", basey=10)
    ax.yaxis.set_ticks([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    # ax.xaxis.set_ticks([1, 1e1, 1e2,1e3,1e4,1e5,1e6])
    ax.set_xlim(1, 1e7)
    #ax.set_ylim(1e-9, 1e-3)

plt.setp(axs[0][1].get_yticklabels(), visible=False)
plt.setp(axs[1][1].get_yticklabels(), visible=False)
plt.setp(axs[0][0].get_xticklabels(), visible=False)
plt.setp(axs[0][1].get_xticklabels(), visible=False)


axs[0][0].set_ylabel("time (s)")
axs[1][0].set_ylabel("time (s)")

axs[1][1].set_xlabel("cardinality")
axs[1][0].set_xlabel("cardinality")

base2_configs = [setsketch1_config, setsketch2_config, ghll_config, ghll_optimized_config, hll_config, hll_optimized_config]

for c in base2_configs:
    make_plot(axs[0][0], data, num_registers=256, base=2, config=c)
fig.text(0.97, 0.97, "$m=256, b=2$", horizontalalignment='right', verticalalignment='top', transform=axs[0][0].transAxes)

for c in base2_configs:
    make_plot(axs[1][0], data, num_registers=4096, base=2, config=c)
fig.text(0.97, 0.97, "$m=4096, b=2$", horizontalalignment='right', verticalalignment='top', transform=axs[1][0].transAxes)

base1_001_configs = [setsketch1_config, setsketch2_config, ghll_config, ghll_optimized_config, minhash_config]

for c in base1_001_configs:
    make_plot(axs[0][1], data, num_registers=256, base=1.001, config=c)
fig.text(0.97, 0.97, "$m=256, b=1.001$", horizontalalignment='right', verticalalignment='top', transform=axs[0][1].transAxes)

for c in base1_001_configs:
    make_plot(axs[1][1], data, num_registers=4096, base=1.001, config=c)
fig.text(0.97, 0.97, "$m=4096, b=1.001$", horizontalalignment='right', verticalalignment='top', transform=axs[1][1].transAxes)

# legend
leg_lines=[]
leg_labels=[]

leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_defs.colorsSketches["SetSketch1"], linestyle="solid"))
leg_labels.append("SetSketch1")

leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_defs.colorsSketches["SetSketch2"], linestyle="solid"))
leg_labels.append("SetSketch2")

leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_defs.colorsSketches["GeneralizedHyperLogLog"], linestyle="solid"))
leg_labels.append("GHLL")

leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_defs.colorsSketches["HyperLogLog"], linestyle="solid"))
leg_labels.append("HLL")

leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_defs.colorsSketches["GeneralizedHyperLogLog"], linestyle="dotted"))
leg_labels.append("GHLL (lower bound tracking)")

leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_defs.colorsSketches["HyperLogLog"], linestyle="dotted"))
leg_labels.append("HLL (lower bound tracking)")

leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_defs.colorsSketches["MinHash"], linestyle="dashdot"))
leg_labels.append("MinHash")

fig.legend(leg_lines, leg_labels, loc="lower center", ncol=4)

fig.subplots_adjust(hspace=0.1, wspace=0.04, top=0.98, bottom=0.29, left=0.09, right=0.995)

fig.savefig('paper/performance.pdf', format='pdf', dpi=1200, metadata={'creationDate': None} )
plt.close(fig)
