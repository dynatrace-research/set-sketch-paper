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

import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator
matplotlib.use("PDF")
import math
import glob

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\RequirePackage[T1]{fontenc} \RequirePackage[tt=false, type1=true]{libertine} \RequirePackage[varqu]{zi4} \RequirePackage[libertine]{newtxmath}')

linewidth_incl_excl = 1
linewidth_new = 1
linewidth_theory = 1
linestyle_incl_excl = "dashed"
linestyle_new = "solid"
linestyle_theory = "dashdot"

def calculateMinHashRelativeRMSE(J, u, m):
    return math.sqrt((1-J)/(J*m))


def calculateSuperMinHashRelativeRMSE(J, u, m):
    s = 0
    for q in range(1, m):
        s += pow(q / (m - 1), u) * (pow((q+1)/m, u) +
                                    pow((q-1)/m, u) - 2*pow(q/m, u))
    factor = 1. - s * (m - 1)/(u - 1)
    return math.sqrt(factor * (1-J)/(J*m))

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

def get_true_union_size(data, j):
    return int(data["trueIntersection"][j]) + int(data["trueDifference1"][j]) + int(data["trueDifference2"][j])

def get_true_jaccard_similarity_inverse(data, j):
    return get_true_union_size(data, j) / int(data["trueIntersection"][j])


def make_chart(colors, data, size, union_cardinality, ax, new_label, incl_excl_label, true_label, intersect_values):
    for idx in range(0, len(intersect_values)):
        i = intersect_values[idx]
        ratios = []
        new_jaccard_relative_errors = []
        incl_excl_jaccard_relative_errors = []
        for j in range(0, size):
            if data["trueIntersection"][j] != i or data["trueDifference2"][j] == 0 or get_true_union_size(data, j) != union_cardinality:
                continue

            ratios.append(data["trueDifference1"][j] / data["trueDifference2"][j])

            new_jaccard_relative_errors.append(math.sqrt(data[new_label][j])/ data[true_label][j])
            incl_excl_jaccard_relative_errors.append(math.sqrt(data[incl_excl_label][j])/ data[true_label][j])

        ax.plot(ratios, new_jaccard_relative_errors, color = colors[idx], linewidth=linewidth_new, linestyle=linestyle_new)
        ax.plot(ratios, incl_excl_jaccard_relative_errors, color = colors[idx], linewidth=linewidth_incl_excl, linestyle=linestyle_incl_excl)

def plot_superminhash_relative_rmse(colors, data, size, union_cardinality, ax, intersect_values, m):
    for idx in range(0, len(intersect_values)):
        j = intersect_values[idx]/union_cardinality
        relative_rmse = calculateSuperMinHashRelativeRMSE(j, union_cardinality, m)
        relative_rmse_mh = calculateMinHashRelativeRMSE(j, union_cardinality, m)
        # ax.plot([1,1e9], [relative_rmse,relative_rmse], color = colors[idx], linewidth=linewidthML, linestyle="dashdot")
        # print("SMH union=" + str(union_cardinality) + " J=" + str(j) + " relRMSE=" + str(relative_rmse) + " " + str(relative_rmse/relative_rmse_mh))

def plot_minhash_relative_rmse(colors, data, size, union_cardinality, ax, intersect_values, m):
    for idx in range(0, len(intersect_values)):
        j = intersect_values[idx]/union_cardinality
        relative_rmse = calculateMinHashRelativeRMSE(j, union_cardinality, m)
        # ax.plot([1,1e9], [relative_rmse,relative_rmse], color = colors[idx], linewidth=linewidthML, linestyle="dashdot")
        # print("MH union=" + str(union_cardinality) + " J=" + str(j) + " relRMSE=" + str(relative_rmse))

def calculate_theoretical_relativ_standard_error(alpha_beta_ratio, jaccard_similarity, m, base_value):
    beta = (1 - jaccard_similarity) / (1 + alpha_beta_ratio)
    alpha = beta * alpha_beta_ratio

    ax = 1 - alpha + alpha / base_value
    bx = 1 - beta + beta / base_value

    logax = math.log(ax, base_value)
    logbx = math.log(bx, base_value)

    factor = -pow(math.log(base_value),2)/(m * math.pow(1-1/base_value,2))

    return math.sqrt(factor * (ax*ax*logax + bx*bx*logbx + pow(ax*logax + bx*logbx,2))) / jaccard_similarity

def plot_theoretical_relativ_standard_error(colors, data, size, union_cardinality, ax, intersect_values, m, base_value):
    b = 1.01
    max_alpha_beta_ratio = 10000
    alpha_beta_ratios = [pow(b, k) for k in range(0, math.ceil(math.log(max_alpha_beta_ratio)/math.log(b)))]

    for idx in range(0, len(intersect_values)):
        intersection_cardinality = intersect_values[idx]
        jaccard_similarity = intersect_values[idx]/union_cardinality
        theoretical_standard_errors = [calculate_theoretical_relativ_standard_error(alpha_beta_ratio, jaccard_similarity, m, base_value) for alpha_beta_ratio in alpha_beta_ratios]
        ax.plot(alpha_beta_ratios, theoretical_standard_errors, color = colors[idx], linewidth=linewidth_theory, linestyle=linestyle_theory)
        # print("MH union=" + str(union_cardinality) + " J=" + str(j) + " relRMSE=" + str(relative_rmse))

def format_union_size(union_size):
    if union_size == 1000:
        return "10^3"
    elif union_size == 1000000:
        return "10^6"
    else:
        assert(False)

def format_jaccard_similarity(jaccard_index_inv):
    if jaccard_index_inv == 10:
        return "0.1"
    if jaccard_index_inv == 100:
        return "0.01"
    if jaccard_index_inv == 1000:
        return "0.001"
    else:
        assert(False)

def make_charts(all_data, sketch_name, union_sizes, jaccard_indices_inv, base_values):

    colors = ["#003f5c", "#bc5090", "#ffa600"]

    fig = plt.figure(figsize=(13, 8), constrained_layout=False)

    outer_grid = fig.add_gridspec(ncols=1, nrows=len(base_values), wspace=0., hspace=0.07)

    inner_grids = []
    for base_value_idx in range(0, len(base_values)):
        inner_grid = outer_grid[base_value_idx,0].subgridspec(ncols=5, nrows=len(union_sizes), wspace=0.04, hspace=0.07)
        inner_grids.append(inner_grid)

    axs = []
    for base_value_idx in range(0, len(base_values)):
        axs1 = []
        axs.append(axs1)
        for union_size_idx in range(0, len(union_sizes)):
            axs2 = []
            axs1.append(axs2)
            for i in range(0,5):
                ax = fig.add_subplot(inner_grids[base_value_idx][union_size_idx, i])
                axs2.append(ax)
                ax.set_xscale("log", basex=10)
                ax.set_yscale("log", basey=10)
                ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0,numticks=100))
                ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0,subs=(0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=100))
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.xaxis.set_ticks([1, 1e1, 1e2, 1e3])
                ax.xaxis.set_ticklabels(["$10^{0}$","$10^{1}$","$10^{2}$",""])
                ax.set_xlim([1, 1e3])
                #ax.set_ylim([3e-3, 5])

                # top labels
                if union_size_idx == 0 and base_value_idx == 0:
                    if i == 0:
                        s = r"Jaccard similarity $J=|A\cap B|/|A\cup B|$"
                    elif i == 1:
                        s = r"cosine similarity $|A\cap B|/\sqrt{|A||B|}$"
                    elif i == 2:
                        s = r"inclusion coefficient $|A\cap B|/|A|$"
                    elif i == 3:
                        s = r"inclusion coefficient $|A\cap B|/|B|$"
                    elif i == 4:
                        s = r"intersection size $\textstyle|A\cap B|$"
                    ax2 = ax.twiny()
                    ax2.set_xticks([])
                    ax2.set_xlabel(s)

                # hide tick labels
                if base_value_idx != len(base_values)-1 or union_size_idx != len(union_sizes)-1:
                    ax.tick_params(labelbottom=False)
                if i!=0:
                    ax.tick_params(labelleft=False)

                # right labels
                if i==0:
                    ax.set_ylabel("relative RMSE")

                # left labels
                if i == 4:
                    ax.text(x=1.04, y=0.5, rotation=270, s="$\mu =" + format_union_size(union_sizes[union_size_idx]) + "$", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

                    if union_size_idx==0:
                        ax.text(x=1.12, y=0.0, rotation=270, s="$b=" + str(base_values[base_value_idx]) + "$", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

                # bottom labels
                if base_value_idx == len(base_values)-1 and union_size_idx == len(union_sizes)-1:
                    ax.set_xlabel(r"$\alpha/\beta$")


    for base_value_idx in range(0, len(base_values)):

        for union_size_idx in range(0, len(union_sizes)):

            base_value = base_values[base_value_idx]
            union_size = union_sizes[union_size_idx]

            for info, data, size in all_data:

                if info["name"] != sketch_name: continue
                if float(info["base"]) != base_value: continue
                if not any([union_size == get_true_union_size(data, j) for j in range(0, size)]): continue
                assert(16384 == int(info["numRegisters"]))
                m = int(info["numRegisters"])

                intersect_values = [union_size // j for j in jaccard_indices_inv]

                make_chart(colors, data, size, union_size, axs[base_value_idx][union_size_idx][0], "newMSEJaccard", "inclExclMSEJaccard","trueJaccard",intersect_values)
                # if sketch_name == "SetSketch2":
                #     plot_superminhash_relative_rmse(colors, data, size, union_size, axs[base_value_idx][union_size_idx][0], intersect_values, m)
                # elif sketch_name == "SetSketch1":
                #     plot_minhash_relative_rmse(colors, data, size, union_size, axs[base_value_idx][union_size_idx][0], intersect_values, m)
                # if sketch_name == "SetSketch1" or sketch_name == "SetSketch2":
                #      plot_theoretical_relativ_standard_error(colors, data, size, union_size, axs[base_value_idx][union_size_idx][0], intersect_values, m, base_value)
                make_chart(colors, data, size, union_size, axs[base_value_idx][union_size_idx][1], "newMSECosine", "inclExclMSECosine","trueCosine",intersect_values)
                make_chart(colors, data, size, union_size, axs[base_value_idx][union_size_idx][2], "newMSEInclusionCoefficient1", "inclExclMSEInclusionCoefficient1","trueInclusionCoefficient1",intersect_values)
                make_chart(colors, data, size, union_size, axs[base_value_idx][union_size_idx][3], "newMSEInclusionCoefficient2", "inclExclMSEInclusionCoefficient2","trueInclusionCoefficient2",intersect_values)
                make_chart(colors, data, size, union_size, axs[base_value_idx][union_size_idx][4], "newMSEIntersection", "inclExclMSEIntersection","trueIntersection",intersect_values)

    # legend
    leg_lines=[]
    leg_labels=[]

    for idx in range(0, len(jaccard_indices_inv)):
        leg_lines.append(matplotlib.lines.Line2D([0], [0], color=colors[idx], lw=1))
        leg_labels.append(r"$J=" + format_jaccard_similarity(jaccard_indices_inv[idx]) + "$")
    leg_lines.append(matplotlib.lines.Line2D([0], [0], color="gray", lw=linewidth_incl_excl, linestyle=linestyle_incl_excl))
    leg_labels.append("inclusion-exclusion principle")
    leg_lines.append(matplotlib.lines.Line2D([0], [0], color="gray", lw=linewidth_new, linestyle=linestyle_new))
    leg_labels.append("new")
    #leg_lines.append( matplotlib.lines.Line2D([0], [0], color="gray", lw=linewidth_theory, linestyle=linestyle_theory))
    #leg_labels.append("theory")

    fig.legend(leg_lines, leg_labels, loc="lower center", ncol=len(leg_labels), bbox_to_anchor=(0.5,-0.005))

    outfile = "joint(" + sketch_name + ").pdf"

    fig.subplots_adjust(left=0.041, bottom=0.09, right=0.972, top=0.975)

    fig.savefig('paper/' + outfile, format='pdf', dpi=1200, metadata={'creationDate': None} )
    plt.close(fig)

filenames = glob.glob("data/joint_test*.csv")
all_data = [read_data(f) for f in filenames]
sketch_names = ["SetSketch1", "SetSketch2", "GeneralizedHyperLogLog"]

union_sizes = sorted(set([get_true_union_size(data, j) for info, data, size in all_data for j in range(0, size) ]))
jaccard_indices_inv = sorted(set([get_true_jaccard_similarity_inverse(data, j) for info, data, size in all_data for j in range(0, size) ]))
base_values = sorted(set([float(info["base"]) for info, data, size in all_data]), reverse= True)

for sketch_name in sketch_names:
    make_charts(all_data, sketch_name, union_sizes, jaccard_indices_inv, base_values)