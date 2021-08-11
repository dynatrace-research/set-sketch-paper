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
from math import sqrt,log,exp
import color_defs

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\RequirePackage[T1]{fontenc} \RequirePackage[tt=false, type1=true]{libertine} \RequirePackage[varqu]{zi4} \RequirePackage[libertine]{newtxmath}')

def expected_relative_error(b):
    if b == 1.:
        return 1
    return sqrt((b+1)/(b-1)*log(b)-1)

def space_efficiency(b, max):
    if b == 1.:
        return float('inf')
    numBits = log(log(max)/log(b))/log(2)
    return expected_relative_error(b) * sqrt(numBits)

def cdf_ghll(base, k):
    if k <= 0:
        return 0
    else:
        return 1 - pow(base, -k)

def pdf_ghll(base, k):
    return cdf_ghll(base, k) - cdf_ghll(base, k-1)

def cdf_setsketch(base, a, k):
    return exp(-a*pow(base, -k))

def pdf_setsketch(base, a, k):
    return cdf_setsketch(base, a, k) - cdf_setsketch(base, a, k-1)

def pdf_hyperminhash(r, k):
    n = 2**r
    return pdf_ghll(2, (k + n - 1) // n) / n

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(6, 1.6)

kVals1 = range(0, 16)
kVals2 = range(0, 61)

ax[0].set_ylabel("probability")
ax[0].set_yscale("log", basey=2)
ax[1].set_yscale("log", basey=2)
ax[0].set_xlim([-0.5,15.5])
ax[1].set_xlim([-0.5, 60.5])
ax[0].yaxis.set_ticks([pow(2,-12),pow(2,-10), pow(2,-8), pow(2,-6), pow(2,-4), pow(2,-2)])
ax[1].yaxis.set_ticks([pow(2,-14),pow(2,-12),pow(2,-10), pow(2,-8), pow(2,-6), pow(2,-4)])
ax[0].set_ylim([pow(2,-10), pow(2,-1)])
ax[1].set_ylim([pow(2,-12), pow(2,-3)])

ax[0].bar([kVal-0.2 for kVal in kVals1], [pdf_ghll(pow(2, 1/2), kVal) for kVal in kVals1], width=0.4, label=r"GHLL $(b=\sqrt{2})$", color=color_defs.colorGHLLDensity)
ax[0].bar([kVal+0.2 for kVal in kVals1], [pdf_hyperminhash(1, kVal) for kVal in kVals1], width = 0.4, label=r"HyperMinHash $(r=1)$", color=color_defs.colorHyperMinHashDensity)
ax[0].legend(loc="upper right")#, prop={'size': 8})

ax[1].bar([kVal-0.2 for kVal in kVals2], [pdf_ghll(pow(2, 1/8), kVal) for kVal in kVals2], width=0.4, label=r"GHLL $(b=\sqrt[8]{2})$", color=color_defs.colorGHLLDensity)
ax[1].bar([kVal+0.2 for kVal in kVals2], [pdf_hyperminhash(3, kVal) for kVal in kVals2], width=0.4, label=r"HyperMinHash $(r=3)$", color=color_defs.colorHyperMinHashDensity)
ax[1].legend(loc="upper right")#, prop={'size': 8})

fig.subplots_adjust(left=0.085, bottom=0.13, right=0.992, top=0.985, wspace=0.15)
fig.savefig('paper/probability_densities.pdf', format='pdf', dpi=1200, metadata={'creationDate': None})
plt.close(fig)
