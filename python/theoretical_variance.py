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

def pb(b,x):
    assert(b >= 1)
    if b == 1:
        return x
    else:
        return -log1p(-x*(b-1) / b) / log(b)

# relative to root mean squared error of minwise hashing
def theoretical_relative_rmse(j, b, nu, nv):
    u = nu / (nu + nv)
    v = nv / (nu + nv)
    p1 = pb(b, max(0, u - v*j))
    p2 = pb(b, max(0, v - u*j))
    assert(p1 >= 0)
    assert(p2 >= 0)
    f = pow((b-1)/(b*log(b)),2) if b > 1 else 1
    term1 = pow(v*pow(b, p1),2)/p1 if p1 > 0 else float('inf')
    term2 = pow(u*pow(b, p2),2)/p2 if p2 > 0 else float('inf')
    term3 = pow(v*pow(b, p1) + u*pow(b, p2), 2) / ( 1 - p1 - p2)
    return sqrt(1. / ((f * (term1 + term2 + term3)) * j * (1 - j)))

colors = ["#003f5c", "#58508d", "#bc5090","#ff6361","#ffa600"]
linestyles = [(0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)),(0, (5, 1)),"solid"]

fig, axs = plt.subplots(1,2, sharey = True)
fig.set_size_inches(6, 2)

bvals = [2, 1.2, 1.05, 1.001, 1]

def plot_chart(ax, ratio):

    assert(ratio <= 1)

    ax.set_ylim(0., 2.0)
    ax.set_xlim(-0.02, 1)
    ax.grid(True)

    ax2 = ax.twiny()
    ax2.set_xticks([])
    if (ratio == 1):
        ax2.set_xlabel("$n_U = n_V$")
    else:
        ax2.set_xlabel("$n_U = " + str(ratio) + "n_V$")

    nu = ratio
    nv = 1

    ax.set_xlabel("$J$")
    xvals = sorted([ratio*pow(0.995, k) for k in range(0, 2000)])
    for b_idx in range(0, len(bvals)):
        b = bvals[b_idx]
        yvals = [theoretical_relative_rmse(j, b, nu, nv) for j in xvals]
        ax.plot(xvals, yvals, label="$b=" + str(b) + "$", color = colors[b_idx], linestyle = linestyles[b_idx])

plot_chart(axs[0], 1)
plot_chart(axs[1], 0.5)

axs[0].set_ylabel(r"$I^{-1/2}(J) / \sqrt{J(1-J)/m}$")

# legend
leg_lines=[]
leg_labels=[]
for k in range(0,len(bvals)):
    leg_lines.append(matplotlib.lines.Line2D([0], [0], lw=1, color=colors[k], linestyle = linestyles[k]))
    leg_labels.append("$b=" + str(bvals[k]) + "$")
fig.legend(leg_lines, leg_labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(leg_labels))

fig.subplots_adjust(left=0.065, bottom=0.36, right=0.985, top=0.915, hspace=0.12, wspace=0.08)

fig.savefig('paper/theoretical_variance.pdf', format='pdf', dpi=1200, metadata={'creationDate': None})
plt.close(fig)