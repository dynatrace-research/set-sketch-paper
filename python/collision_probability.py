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
def estimateJLowerBound(c,m,b):
    p = c/m
    l = log1pdiv(b-1)
    x = expm1div((p+1)*l*(b-1)/2)
    xl = x*l
    return xl*p+(xl-1)

# (b^(c/m)-1)/(b-1)
def estimateJUpperBound(c,m,b):
    p = c/m
    l = log1pdiv(b-1)
    lp = l*p
    return expm1div((b-1)*lp)*lp

def prob(j,b, factor):
    y = j+(1-j)*factor*(1-j)*(b-1)/b
    return log1pdiv((b-1)*y) * y / log1pdiv(b-1)

# log_b(1+J*(b-1))
def probLowerBound(j,b):
    return prob(j,b,0)

# log_b(1+J*(b-1)+(1-J)^2(b-1)^2/(4b))
def probUpperBound(j,b):
    return prob(j,b,0.25)

def theoreticalMseMinHash(j,m):
    return j*(1-j)/m

def theoreticalMseUpperBoundEstimator(j, p, m, b):
    l = log1pdiv(b-1)
    k = l * (b-1) / m
    g1 = expm1div(k)
    g2 = expm1div(2*k)
    h1 = log1pdiv(p*g1*k) * g1*l
    h2 = log1pdiv(p*g2*2*k)*g2 * l
    z1 = expm1div(p*(b-1)*h1) * h1
    z2 = expm1div(p*2*(b-1)*h2) * h2

    mse = 2*p*(z2-z1)/(b-1) - j*(2*p*z1-j)
    return mse

def maxMseUpperBoundEstimate(j,m,b):
    pUpperBound = probUpperBound(j,b)
    pLowerBound = probLowerBound(j,b)

    result = 0.
    for p in numpy.linspace(pLowerBound,pUpperBound,100):
        result = max(result, theoreticalMseUpperBoundEstimator(j, p, m, b))
    return result

def empiricalMseUpperBoundEstimator(j, p, m, b):  
    mse = 0
    for c in range(0,m+1):
        prob = binom.pmf(c,m,p)
        est = estimateJUpperBound(c,m,b)
        mse += prob * pow(est-j,2)
    return mse

def makeMSEUpperboundEstimationChart():
    colors = ["#ffa600","#ff6361","#bc5090","#58508d","#003f5c"]

    fig, axs = plt.subplots(1,2, sharey = True)
    fig.set_size_inches(6, 2)

    bvals = [2, 1.2, 1.08, 1.02, 1.001]
    mvals = [256, 4096]

    for i in range(0,len(mvals)):
        m = mvals[i]
        ax = axs[i]
        for bIdx in range(0, len(bvals)):
            b = bvals[bIdx]
            x_theoretical = []
            y_theoretical = []
            for j in numpy.linspace(1e-6,1-1e-3,1000):
                p = probUpperBound(j,b)
                mse1 = theoreticalMseUpperBoundEstimator(j, p, m, b)
                mse2 = theoreticalMseMinHash(j,m)
                x_theoretical.append(j)
                y_theoretical.append(sqrt(mse1/mse2))
            pl = ax.plot(x_theoretical, y_theoretical, label="$b=" + str(b) + "$", color = colors[bIdx])

            # x_empirical = []
            # y_empirical = []
            # for j in numpy.linspace(1e-3,1-1e-3,10):
            #     p = probUpperBound(j,b)
            #     mse1 = empiricalMseUpperBoundEstimator(j, p, m, b)
            #     mse2 = theoreticalMseMinHash(j,m)
            #     x_empirical.append(j)
            #     y_empirical.append(sqrt(mse1/mse2))

            # ax.scatter(x_empirical, y_empirical, marker='.', color=pl[0].get_color())

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
        leg_lines.append(matplotlib.lines.Line2D([0], [0], lw=1, color=colors[k] ))
        leg_labels.append("$b=" + str(bvals[k]) + "$")
    fig.legend(leg_lines, leg_labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(leg_labels))
    
    fig.subplots_adjust(left=0.082, bottom=0.36, right=0.985, top=0.925, hspace=0.1, wspace=0.08)
    
    axs[0].set_ylabel(r"$\text{RMSE}/\sqrt{J(1-J)/m}$")

    # ax.plot(xVals, [(pow(base, x)-1)/(base -1)-x for x in xVals], label="upper bound")

    fig.savefig('paper/mse_upperbound_estimation.pdf', format='pdf', dpi=1200, metadata={'creationDate': None})
    plt.close(fig)

def makeCollisionProbabilityChart():
    fig, axs = plt.subplots(1,2, sharey = True)
    fig.set_size_inches(6, 1.8)

    bvals = [2, 1.2]
    jvals = numpy.linspace(0,1,1000)

    for i in range(0,len(bvals)):
        b = bvals[i]
        ax = axs[i]
        ax.fill_between(jvals, [probLowerBound(j,b) for j in jvals],[probUpperBound(j,b) for j in jvals], edgecolor="black", facecolor=color_defs.colorBackgroundGray, zorder=100)
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.yaxis.set_ticks([0.0,0.2,0.4,0.6,0.8,1.0])
        ax.set_xlabel("$J$")
        ax2 = ax.twiny()
        ax2.set_xticks([])
        ax2.set_xlabel("$b=" + str(b) + "$")


    axs[0].set_ylabel("$\Pr(K_{Ai}=K_{Bi})$")
    
    fig.subplots_adjust(left=0.08, bottom=0.225, right=0.985, top=0.91, hspace=0.1, wspace=0.1)

    fig.savefig('paper/collision_probability.pdf', format='pdf', dpi=1200, metadata={'creationDate': None})
    plt.close(fig)

makeCollisionProbabilityChart()
makeMSEUpperboundEstimationChart()