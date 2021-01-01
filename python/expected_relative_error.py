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
from math import sqrt,log

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


bases = numpy.linspace(1,10,10000)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(6, 2)

ax0 = ax

ax0.set_ylim([0.975, 1.375])
ax0.set_xlim([1,10])

ax0.set_xlabel("$b$")
ax0.set_ylabel(r"$\operatorname{RSD}\times\sqrt{m}$")

ax0.plot(bases, [expected_relative_error(b) for b in bases], color="black")

ax0.grid(True)
ax0.yaxis.set_ticks(numpy.arange(1.0, 1.35,0.05))
ax0.xaxis.set_ticks(numpy.arange(1,11,1))

fig.subplots_adjust(hspace=0.1, wspace=0.04, top=0.993, bottom=0.195, left=0.088, right=0.99)

fig.savefig('paper/expected_relative_error.pdf', format='pdf', dpi=1200, metadata={'creationDate': None})
plt.close(fig)
