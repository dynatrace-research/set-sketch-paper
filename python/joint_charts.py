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
import numpy

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\RequirePackage[T1]{fontenc} \RequirePackage[tt=false, type1=true]{libertine} \RequirePackage[varqu]{zi4} \RequirePackage[libertine]{newtxmath}')

linewidth_incl_excl = 1
linestyle_incl_excl = "dashed"
color_incl_excl = "blue"

linewidth_new = 1
linestyle_new = "solid"
color_new = "black"

linewidth_new_known_card = 2
linestyle_new_known_card = "dotted"
color_new_known_card="black"

linewidth_theory = 1
linestyle_theory = "solid"
color_theory = "red"

linewidth_simple_deprecated = 1
linestyle_simple_deprecated = "dashdot"
color_simple_deprecated = "blue"

linewidth_original = 1
linestyle_original = "solid"
color_original = "gray"

linewidth_original_known_card = 2
linestyle_original_known_card = "dotted"
color_original_known_card = "gray"

def calculate_minhash_relative_rmse(j, m):
    return math.sqrt((1-j)/(j*m))


def calculate_superminhash_relative_rmse(j, u, m):
    s = 0
    for q in range(1, m):
        s += pow(q / (m - 1), u) * (pow((q+1)/m, u) +
                                    pow((q-1)/m, u) - 2*pow(q/m, u))
    factor = 1. - s * (m - 1)/(u - 1)
    return math.sqrt(factor * (1-j)/(j*m))

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

def get_true_jaccard_similarity(data, j):
    return int(data["trueIntersection"][j]) / get_true_union_size(data, j)

def p(b,x):
    if b > 1:
        return -math.log1p(-x*(b-1) / b) / math.log(b)
    else:
        return x

def calculate_theoretical_relative_standard_error_jaccard(alpha_beta_ratio, j, m, b):
    u = (alpha_beta_ratio + j) / (1 + alpha_beta_ratio)
    v = (1 + j * alpha_beta_ratio) / (1 + alpha_beta_ratio)
    uu = u / (u + v)
    vv = v / (u + v)
    factor = (b-1)/(b * math.log(b)) if b > 1 else 1
    pu = p(b, uu - vv * j)
    pv = p(b, vv - uu * j)
    bu = pow(b, pu)
    bv = pow(b, pv)
    fisher_info =  m * pow(factor,2) * (pow(vv*bu + uu*bv,2) / (1 - pu - pv)  + pow(vv*bu,2) / pu + pow(uu*bv,2) / pv)
    return math.sqrt(1/fisher_info) / j

def calculate_theoretical_relative_standard_error_difference1(alpha_beta_ratio, j, m, b):
    u = (alpha_beta_ratio + j) / (1 + alpha_beta_ratio)
    v = (1 + j * alpha_beta_ratio) / (1 + alpha_beta_ratio)
    return calculate_theoretical_relative_standard_error_jaccard(alpha_beta_ratio, j, m, b) * (u+v)/(u-v*j) * j/(1+j)

def expected_relative_rmse_jaccard(ratios, b, m, j):
    return [calculate_theoretical_relative_standard_error_jaccard(r, j, m, b) for r in ratios]

def original_relative_rmse_jaccard(ratios, m, j):
    return [calculate_minhash_relative_rmse(j, m) for _ in ratios]

def expected_relative_rmse_intersection(ratios, b, m, j):
    return [calculate_theoretical_relative_standard_error_jaccard(r, j, m, b) / (1+j) for r in ratios]

def expected_relative_rmse_difference1(ratios, b, m, j):
    return [calculate_theoretical_relative_standard_error_difference1(r, j, m, b) for r in ratios]


def extract_relative_error(matching_data, prefix, postfix, jaccard, union_size):
    info, data, size = matching_data

    ratios = []
    values = []

    postfix1 = postfix
    if postfix1[-1] == "1":
        postfix1 = postfix1[:-1] + "2"
    elif  postfix1[-1] == "2":
        postfix1 = postfix1[:-1] + "1"
    postfix2 = postfix

    for j in range(size-1, 0, -1):
        if get_true_jaccard_similarity(data, j) != jaccard or data["trueDifference1"][j] == 0  or data["true" + postfix1][j] == 0 or get_true_union_size(data, j) != union_size:
            continue
        ratios.append(data["trueDifference2"][j] / data["trueDifference1"][j])
        values.append(math.sqrt(data[prefix + "MSE" + postfix1][j])/ data["true" + postfix1][j])

    for j in range(0, size):
        if get_true_jaccard_similarity(data, j) != jaccard or data["trueDifference2"][j] == 0  or data["true" + postfix2][j] == 0 or get_true_union_size(data, j) != union_size:
            continue
        ratios.append(data["trueDifference1"][j] / data["trueDifference2"][j])
        values.append(math.sqrt(data[prefix + "MSE" + postfix2][j])/ data["true" + postfix2][j])

    return ratios,values

def plot_charts(ax, all_data, b, union_size, jaccard, m, sketch_name):

    matching_data = None
    for selected_data in all_data:
        info, data, size = selected_data
        if float(info["base"]) != b: continue
        if int(info["numRegisters"]) != m: continue
        if info["name"] != sketch_name: continue
        matching_data = selected_data
        break

    assert(matching_data != None)

    x = numpy.logspace(-3, 3, 1000)
    for a,f in zip(ax, [expected_relative_rmse_jaccard, expected_relative_rmse_intersection, expected_relative_rmse_intersection, expected_relative_rmse_intersection, expected_relative_rmse_difference1]):
        a.plot(x, f(x, b, m, jaccard), color=color_theory, linestyle=linestyle_theory, linewidth=linewidth_theory)

    for a,s in zip(ax, ["Jaccard", "Cosine", "InclusionCoefficient1", "Intersection", "Difference1"]):
        if sketch_name == "MinHash" or sketch_name == "HyperMinHash":
            x,y = extract_relative_error(matching_data, "original", s, jaccard, union_size)
            a.plot(x, y, color=color_original, linestyle=linestyle_original, linewidth=linewidth_original)
            x,y = extract_relative_error(matching_data, "originalKnownCard", s, jaccard, union_size)
            a.plot(x, y, color=color_original_known_card, linestyle=linestyle_original_known_card, linewidth=linewidth_original_known_card)
        x,y = extract_relative_error(matching_data, "new", s, jaccard, union_size)
        a.plot(x, y, color=color_new, linestyle=linestyle_new, linewidth=linewidth_new)
        x,y = extract_relative_error(matching_data, "newKnownCard", s, jaccard, union_size)
        a.plot(x, y, color=color_new_known_card, linestyle=linestyle_new_known_card, linewidth=linewidth_new_known_card)
        x,y = extract_relative_error(matching_data, "inclExcl", s, jaccard, union_size)
        a.plot(x, y, color=color_incl_excl, linestyle=linestyle_incl_excl, linewidth=linewidth_incl_excl)

    for a in ax:
        a.set_xscale("log", basex=10)
        a.set_xlim(1e-3,1e3)
        a.set_yscale("log", basey=10)

def draw_figure(sketch_name, all_data, union_size, b_values, fig):

    num_charts = 5
    j_values = [0.01, 0.1, 0.5]

    outer_grid = fig.add_gridspec(ncols=1, nrows=len(b_values), wspace=0., hspace=0.08)
    inner_grids = [outer_grid[i, 0].subgridspec(ncols=5, nrows=len(j_values), wspace=0.2, hspace=0.08) for i in range(0,len(b_values))]

    axs = []
    for section_idx in range(0,len(b_values)):
        for row_idx in range(0,len(j_values)):
            a = []
            for col_idx in range(0,num_charts):
                a.append(fig.add_subplot(inner_grids[section_idx][row_idx, col_idx]))
            axs.append(a)

    # share x-axes
    ax_list = [a for aa in axs for a in aa ]
    ax_list[0].get_shared_x_axes().join(*ax_list)

    # hide inner x-axis labels
    for i in range(0, len(axs)):
        if i % len(j_values) != 2:
            for aa in axs[i]:
                aa.tick_params(labelbottom=False)

    # share y-axes
    for k in range(0,len(j_values)):
        ax_list = []
        for i in range(0, len(b_values)):
            ax_list += axs[k+i*len(j_values)][:-1]
        ax_list[0].get_shared_y_axes().join(*ax_list)
        ax_list = []
        for i in range(0, len(b_values)):
            ax_list.append(axs[k+i*len(j_values)][-1])
        ax_list[0].get_shared_y_axes().join(*ax_list)


    for i in range(0,len(j_values)):
        for j in range(0,len(b_values)):
            plot_charts(axs[len(j_values)*j + i], all_data, b_values[j], union_size, j_values[i], 4096, sketch_name)


    # set y-scales
    axs[0][0].set_ylim([4e-2,1.5e0])
    axs[1][0].set_ylim([4e-3,1.5e-1])
    axs[2][0].set_ylim([5e-4,4e-2])
    for a in axs:
        a[-1].set_ylim([3e-4,1.5e0])
        a[-1].set_yticks([1e-3, 1e-2, 1e-1, 1e0])

    # set x-ticks
    for a in axs:
        for aa in a:
            aa.set_xticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])

    # top labels
    top_labels = [r"Jaccard similarity $J=|U\cap V|/|U\cup V|$", r"cosine similarity $|U\cap V|/\sqrt{|U||V|}$", r"inclusion coefficient $|U\cap V|/|U|$", r"intersection size $|U\cap V|$", r"difference size $\textstyle|U\setminus V|$"]
    for i in range(0, num_charts):
        ax2 = axs[0][i].twiny()
        ax2.set_xticks([])
        ax2.set_xlabel(top_labels[i])

    # bottom labels
    for axx in axs[-1]:
        axx.set_xlabel(r"$|U\setminus V|/|V\setminus U|$")

    # left labels
    for axx in axs:
        fig.text(x=-0.22, y=0.5, rotation=90, s="relative RMSE", horizontalalignment='center', verticalalignment='center', transform=axx[0].transAxes)

    # right labels
    for i in range(0, len(b_values)):
        for a,j in zip([axs[k] for k in range(i*len(j_values), (i+1)*len(j_values))], j_values):
            fig.text(x=1.04, y=0.5, rotation=270, s="$J=" + str(j) + "$" , horizontalalignment='center', verticalalignment='center', transform=a[-1].transAxes)
        b = b_values[i]
        if len(b_values) > 1:
            b_str = str(b) if b != 1.00067713069306641e+00 else r"2^{-2^{10}}"
            fig.text(x=1.12, y=0.5, rotation=270, s="$b=" + b_str + "$" , horizontalalignment='center', verticalalignment='center', transform=axs[len(j_values)*i + 1][-1].transAxes)

    # legend
    leg_lines=[]
    leg_labels=[]

    leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_new, lw=linewidth_new, linestyle=linestyle_new))
    leg_labels.append("new")
    leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_new_known_card, lw=linewidth_new_known_card, linestyle=linestyle_new_known_card))
    leg_labels.append("new (cardinalities known)")
    leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_theory, lw=linewidth_theory, linestyle=linestyle_theory))
    leg_labels.append("theory new (cardinalities known)")

    if sketch_name == "MinHash" or sketch_name == "HyperMinHash":
        leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_original, lw=linewidth_original, linestyle=linestyle_original))
        leg_labels.append("original")
        leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_original_known_card, lw=linewidth_original_known_card, linestyle=linestyle_original_known_card))
        leg_labels.append("original (cardinalities known)")
    leg_lines.append(matplotlib.lines.Line2D([0], [0], color=color_incl_excl, lw=linewidth_incl_excl, linestyle=linestyle_incl_excl))
    leg_labels.append("inclusion-exclusion")


    fig.legend(leg_lines, leg_labels, loc="lower center", ncol=len(leg_labels), bbox_to_anchor=(0.5,-0.005))

def create_figure(sketch_name, all_data, union_size):

    b_values = [2, 1.001]

    fig = plt.figure(figsize=(13, 6.95), constrained_layout=False)

    draw_figure(sketch_name, all_data, union_size, b_values, fig)

    fig.subplots_adjust(left=0.041, bottom=0.105, right=0.973, top=0.97)

    outfile = "joint_" + sketch_name + "_" + str(union_size) + ".pdf"
    fig.savefig('paper/' + outfile, format='pdf', dpi=1200, metadata={'creationDate': None} )
    plt.close(fig)

def create_small_figure(sketch_name, all_data, union_size, b):

    b_values = [b]

    fig = plt.figure(figsize=(13, 3.9), constrained_layout=False)

    draw_figure(sketch_name, all_data, union_size, b_values, fig)

    fig.subplots_adjust(left=0.041, bottom=0.185, right=0.973, top=0.952)

    outfile = "joint_" + sketch_name + "_" + str(union_size) + "_" + str(b).replace(".","_") + ".pdf"
    fig.savefig('paper/' + outfile, format='pdf', dpi=1200, metadata={'creationDate': None} )
    plt.close(fig)


filenames = glob.glob("data/joint_test*.csv")
all_data = [read_data(f) for f in filenames]

union_sizes = [1000, 1000000]

for sketch_name in ["SetSketch1", "SetSketch2", "GeneralizedHyperLogLog"]:
    for union_size in union_sizes:
        create_figure(sketch_name, all_data, union_size)

for union_size in union_sizes:
    create_small_figure("MinHash", all_data, union_size, 1.)

# for sketch_name in ["SetSketch1", "SetSketch2", "GeneralizedHyperLogLog"]:
#     for union_size in union_sizes:
#         create_small_figure(sketch_name, all_data, union_size, 2)
#         create_small_figure(sketch_name, all_data, union_size, 1.001)

for union_size in union_sizes:
    create_small_figure("HyperMinHash", all_data, union_size, 1.00067713069306641e+00)

