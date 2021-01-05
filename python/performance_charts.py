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

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\RequirePackage[T1]{fontenc} \RequirePackage[tt=false, type1=true]{libertine} \RequirePackage[varqu]{zi4} \RequirePackage[libertine]{newtxmath}')


sketch_names = ["SetSketch1", "SetSketch2", "GeneralizedHyperLogLog", "HyperLogLog"]
sketch_labels = {"SetSketch1":"SetSketch1", "SetSketch2":"SetSketch2", "GeneralizedHyperLogLog":"GHLL", "HyperLogLog":"HLL"}
linestyles = {"SetSketch1" : "solid","SetSketch2" : "dashed","GeneralizedHyperLogLog" :"dashdot","HyperLogLog" : (0,(1,1))}


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
    cardinalities = []
    times = []
    for d in data:
        if not "dummy" in d[0]:
            continue
        dd = d[1]
        ax.plot(dd["cardinality"], dd["avg time in seconds"])


def make_plot(ax, data, num_registers, base, sketch_name):
    cardinalities = []
    times = []
    for d in data:
        if not "numRegisters" in d[0] or not "base" in d[0] or not "name" in d[0]or int(d[0]["numRegisters"]) != num_registers or float(d[0]["base"]) != base or d[0]["name"] != sketch_name:
            continue
        dd = d[1]
        ax.plot(dd["cardinality"], dd["avg time in seconds"],linestyle=linestyles[sketch_name], color=color_defs.colorsSketches[sketch_name])


filenames = glob.glob("data/performance_test*.csv")

data = []
for filename in filenames:
    data.append(readData(filename))

fig, axs = plt.subplots(2, 2)
fig.set_size_inches(6,3)

for ax in fig.axes:
    ax.set_xscale("log", basex=10)
    ax.set_yscale("log", basey=10)
    ax.yaxis.set_ticks([1e-6, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    ax.xaxis.set_ticks([1, 1e1, 1e2,1e3,1e4,1e5,1e6])
    ax.set_xlim(1, 1e7)
    ax.set_ylim(1e-6, 3e-1)

plt.setp(axs[0][1].get_yticklabels(), visible=False)
plt.setp(axs[1][1].get_yticklabels(), visible=False)
plt.setp(axs[0][0].get_xticklabels(), visible=False)
plt.setp(axs[0][1].get_xticklabels(), visible=False)


axs[0][0].set_ylabel("time (s)")
axs[1][0].set_ylabel("time (s)")

# axs[1][1].set_xlabel("cardinality")
# axs[1][0].set_xlabel("cardinality")
axs[1][1].set_xlabel("$n$")
axs[1][0].set_xlabel("$n$")


make_plot(axs[0][0], data, num_registers=256, base=2, sketch_name="SetSketch1")
make_plot(axs[0][0], data, num_registers=256, base=2, sketch_name="SetSketch2")
make_plot(axs[0][0], data, num_registers=256, base=2, sketch_name="HyperLogLog")
make_plot(axs[0][0], data, num_registers=256, base=2, sketch_name="GeneralizedHyperLogLog")
# make_dummy_plot(axs[0][0], data)
fig.text(0.03, 0.97, "$m=256, b=2$", horizontalalignment='left', verticalalignment='top', transform=axs[0][0].transAxes)

make_plot(axs[0][1], data, num_registers=4096, base=2, sketch_name="SetSketch1")
make_plot(axs[0][1], data, num_registers=4096, base=2, sketch_name="SetSketch2")
make_plot(axs[0][1], data, num_registers=4096, base=2, sketch_name="HyperLogLog")
make_plot(axs[0][1], data, num_registers=4096, base=2, sketch_name="GeneralizedHyperLogLog")
# make_dummy_plot(axs[0][1], data)
fig.text(0.03, 0.97, "$m=4096, b=2$", horizontalalignment='left', verticalalignment='top', transform=axs[0][1].transAxes)

make_plot(axs[1][0], data, num_registers=256, base=1.001, sketch_name="SetSketch1")
make_plot(axs[1][0], data, num_registers=256, base=1.001, sketch_name="SetSketch2")
make_plot(axs[1][0], data, num_registers=256, base=1.001, sketch_name="GeneralizedHyperLogLog")
# make_dummy_plot(axs[1][0], data)
fig.text(0.03, 0.97, "$m=256, b=1.001$", horizontalalignment='left', verticalalignment='top', transform=axs[1][0].transAxes)

make_plot(axs[1][1], data, num_registers=4096, base=1.001, sketch_name="SetSketch1")
make_plot(axs[1][1], data, num_registers=4096, base=1.001, sketch_name="SetSketch2")
make_plot(axs[1][1], data, num_registers=4096, base=1.001, sketch_name="GeneralizedHyperLogLog")
# make_dummy_plot(axs[1][1], data)
fig.text(0.03, 0.97, "$m=4096, b=1.001$", horizontalalignment='left', verticalalignment='top', transform=axs[1][1].transAxes)

# legend
leg_lines=[]
leg_labels=[]
for sketch_name in sketch_names:
    leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_defs.colorsSketches[sketch_name], linestyle=linestyles[sketch_name]))
    leg_labels.append(sketch_labels[sketch_name])

fig.legend(leg_lines, leg_labels, loc="lower center", ncol=len(leg_labels))

fig.subplots_adjust(hspace=0.1, wspace=0.04, top=0.994, bottom=0.25, left=0.09, right=0.995)

fig.savefig('paper/performance.pdf', format='pdf', dpi=1200, metadata={'creationDate': None} )
plt.close(fig)
