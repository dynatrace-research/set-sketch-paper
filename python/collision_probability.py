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

import numpy
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
from math import exp, expm1,log,log1p,sqrt
from scipy.optimize import minimize_scalar, minimize, Bounds
from scipy.stats import binom
import color_defs

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\RequirePackage[T1]{fontenc} \RequirePackage[tt=false, type1=true]{libertine} \RequirePackage[varqu]{zi4} \RequirePackage[libertine]{newtxmath}')

# (e^x-1) / x
def expm1div(x):
    return expm1(x)/x if x != 0 else 1

# ln(1+x) / x
def log1pdiv(x):
    return log1p(x)/x if x != 0 else 1

# 2*(b^((c/m+1)/2)-1)/(b-1)-1
def estimate_j_lower_bound(c,m,b):
    p = c/m
    l = log1pdiv(b-1)
    x = expm1div((p+1)*l*(b-1)/2)
    xl = x*l
    return xl*p+(xl-1)

# (b^(c/m)-1)/(b-1)
def estimate_j_upper_bound(c,m,b):
    return (pow(b, c/m) - 1) / (b-1)
    # p = c/m
    # l = log1pdiv(b-1)
    # lp = l*p
    # return expm1div((b-1)*lp)*lp

def prob_lower_bound(j,b):
    return log(1+j*(b-1))/log(b)

def prob_upper_bound(j,b):
    return log(1+j*(b-1)+pow(1-j,2)*pow(b-1,2)/(4*b))/log(b)

def theoretical_mse_minhash(j,m):
    return j*(1-j)/m

def mse_of_upper_bound_estimator_for_equal_set_sizes(j, m, b):
    collision_probability = prob_upper_bound(j,b)
    mse = 0
    for c in range(0,m+1):
        prob = binom.pmf(c,m,collision_probability)
        est = estimate_j_upper_bound(c,m,b)
        mse += prob * pow(est-j,2)
    return mse

def mse_of_upper_bound_estimator_for_equal_set_sizes_fast_explicit_calculation(j, m, b):
    collision_probability = prob_upper_bound(j,b)
    l = log1pdiv(b-1)
    k = l * (b-1) / m
    g1 = expm1div(k)
    g2 = expm1div(2*k)
    h1 = log1pdiv(collision_probability*g1*k) * g1*l
    h2 = log1pdiv(collision_probability*g2*2*k)*g2 * l
    z1 = expm1div(collision_probability*(b-1)*h1) * h1
    z2 = expm1div(collision_probability*2*(b-1)*h2) * h2

    mse = 2*collision_probability*(z2-z1)/(b-1) - j*(2*collision_probability*z1-j)
    return mse

def make_mse_upper_bound_estimation_chart():
    colors = ["#003f5c", "#58508d", "#bc5090","#ff6361","#ffa600"]
    linestyles = [(0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)),(0, (5, 1)),"solid"]

    fig, axs = plt.subplots(1,2, sharey = True)
    fig.set_size_inches(6, 2)

    bvals = [2, 1.2, 1.08, 1.02, 1.001]
    mvals = [256, 4096]

    for i in range(0,len(mvals)):
        m = mvals[i]
        ax = axs[i]
        for b_idx in range(0, len(bvals)):
            b = bvals[b_idx]
            x_theoretical = []
            y_theoretical = []
            for j in numpy.logspace(-1e-3,1-6,200):
                
                #mse1 = mse_of_upper_bound_estimator_for_equal_set_sizes_fast_explicit_calculation(j, m, b)
                mse1 = mse_of_upper_bound_estimator_for_equal_set_sizes(j, m, b)
                mse2 = theoretical_mse_minhash(j,m)
                x_theoretical.append(j)
                y_theoretical.append(sqrt(mse1/mse2))
                if y_theoretical[-1] > 3:
                    break
            pl = ax.plot(x_theoretical, y_theoretical, label="$b=" + str(b) + "$", color = colors[b_idx], linestyle = linestyles[b_idx])

        ax.grid(True)
        ax.set_xlim(-0.02, 1)
        ax.set_ylim(0.95,2.05)
        ax.yaxis.set_ticks([1.0,1.2,1.4,1.6,1.8,2.0])
        ax.set_xlabel("$J$")
        ax2 = ax.twiny()
        ax2.set_xticks([])
        ax2.set_xlabel("$m=" + str(m) + "$")

    # legend
    leg_lines=[]
    leg_labels=[]
    for k in range(0,len(bvals)):
        leg_lines.append(matplotlib.lines.Line2D([0], [0], lw=1, color=colors[k], linestyle = linestyles[k] ))
        leg_labels.append("$b=" + str(bvals[k]) + "$")
    fig.legend(leg_lines, leg_labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(leg_labels))

    fig.subplots_adjust(left=0.082, bottom=0.36, right=0.985, top=0.925, hspace=0.1, wspace=0.08)

    axs[0].set_ylabel(r"$\text{RMSE}/\sqrt{J(1-J)/m}$")

    fig.savefig('paper/mse_upperbound_estimation.pdf', format='pdf', dpi=1200, metadata={'creationDate': None})
    plt.close(fig)

def make_collision_probability_chart():
    fig, axs = plt.subplots(1,3, sharex = True, sharey = True)
    fig.set_size_inches(6, 1.8)

    bvals = [2, 1.2, 1.001]
    jvals = numpy.linspace(0,1,1000)

    for i in range(0,len(bvals)):
        b = bvals[i]
        ax = axs[i]
        ax.fill_between(jvals, [prob_lower_bound(j,b) for j in jvals],[prob_upper_bound(j,b) for j in jvals], edgecolor="black", facecolor=color_defs.colorBackgroundGray, zorder=100)
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_ticks([0.0,0.2,0.4,0.6,0.8,1.0])
        ax.yaxis.set_ticks([0.0,0.2,0.4,0.6,0.8,1.0])
        ax.set_xlabel("$J$")
        ax2 = ax.twiny()
        ax2.set_xticks([])
        ax2.set_xlabel("$b=" + str(b) + "$")

    axs[0].set_ylabel("$P(K_{Ui}=K_{Vi})$")

    fig.subplots_adjust(left=0.08, bottom=0.225, right=0.985, top=0.91, hspace=0.1, wspace=0.14)

    fig.savefig('paper/collision_probability.pdf', format='pdf', dpi=1200, metadata={'creationDate': None})
    plt.close(fig)

make_collision_probability_chart()
make_mse_upper_bound_estimation_chart()