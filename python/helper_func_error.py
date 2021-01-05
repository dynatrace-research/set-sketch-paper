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
import math
from scipy.special import gamma, sinc
from scipy.optimize import minimize_scalar, minimize, Bounds
import cmath

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\RequirePackage[T1]{fontenc} \RequirePackage[tt=false, type1=true]{libertine} \RequirePackage[varqu]{zi4} \RequirePackage[libertine]{newtxmath}')

# calculates (xi_b^r(x) - 1)^2 using the corresponding Fourier series
def xi_squared_error(x,b,r):
    sum = 0
    l = 1
    c1 = -2. * numpy.pi / math.log(b)
    c2 = 2. * numpy.pi *x
    while True:
        oldsum = sum
        sum += gamma(complex(r, c1*l)) * cmath.exp(complex(0, l * c2))
        l += 1
        if oldsum == sum:
            break
    return pow(2./gamma(r)*sum.real,2)

def max_amplitude_xi(b,r):
    max = 0
    for start in numpy.linspace(0,1,100):
        result = minimize_scalar(lambda z: -xi_squared_error(z, b, r), bounds=(0, start, 1), method='Brent', tol= 1e-18)
        assert(result.success)
        if(-result.fun > max):
            max = -result.fun
    return math.sqrt(max)

gamma_cache = {}

# calculates ((zeta_b(x1,x2) - (x2-x1)) / (x2-x1))^2 using the corresponding Fourier series
def zeta_squared_error(x1,x2,b):
    sum = 0
    l = 1
    c1 = -2. * numpy.pi / math.log(b)
    f = cmath.exp(complex(0,numpy.pi*(x1+x2)))
    fn = f
    while True:
        oldsum = sum
        g = gamma_cache.get((c1,l))
        if g is None:
            g = gamma(complex(1, c1*l))
            gamma_cache[(c1,l)] =  g
        sum += g * fn * sinc(l*(x1-x2))
        if oldsum == sum:
            break
        l += 1
        fn *= f
    return pow(sum.real,2)

def max_amplitude_zeta(b):
    max = 0
    for startX in numpy.linspace(0,1,20):
        for startY in numpy.linspace(0,1,20):
            result = minimize(
                lambda z: -zeta_squared_error(z[0], z[1], b),
                x0 = (startX, startY), method="L-BFGS-B",
                bounds = Bounds([0,0],[1,1]))
            if not result.success:
                print(b)
                print(startX)
                print(startY)
                print(result)
                assert(result.success)

            if(-result.fun > max):
                max = -result.fun
    return math.sqrt(max)


bases = numpy.linspace(1.2,5,100)

fig, ax = plt.subplots(1, 2, sharey = True)
fig.set_size_inches(6, 2)

ax[0].set_yscale("log", basey=10)
ax[0].set_ylim([1e-11, 1e0])
ax[0].set_xlim([1,5])
ax[0].set_yticks([1e-11,1e-8,1e-5,1e-2])
ax[0].set_ylabel("relative approx. error")

ax[0].set_xlabel(r"$b$")
ax[0].set_title(r"$\max_x |\xi_b^r(x)-1|$")

ax[0].plot(bases, [max_amplitude_xi(b, 1) for b in bases], label = "$r=1$", color="black")
ax[0].plot(bases, [max_amplitude_xi(b, 2) for b in bases], label = "$r=2$", linestyle="dashed", color="black")
ax[0].grid(True)

leg = ax[0].legend(loc = "lower right")

ax[1].set_xlim([1,5])
ax[1].set_xlabel(r"$b$")

ax[1].plot(bases, [max_amplitude_zeta(b) for b in bases], color="black")

ax[1].set_title(r"$\max_{x_1,x_2} \left|\frac{\zeta_b(x_1,x_2)-(x_2-x_1)}{x_2-x_1}\right|$")
ax[1].grid(True)

fig.subplots_adjust(hspace=0.1, wspace=0.04, top=0.815, bottom=0.195, left=0.096, right=0.991)

fig.savefig('paper/helper_func_error.pdf', format='pdf', dpi=1200, metadata={'creationDate': None})
plt.close(fig)
