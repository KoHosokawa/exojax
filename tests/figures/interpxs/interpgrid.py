"""
Error of grid-based interpolation
"""


import matplotlib.pyplot as plt
import numpy
from exojax.spec.autospec import AutoXS
import pytest
import jax.numpy as jnp
import numpy as np

nuslog = numpy.logspace(numpy.log10(1900.0), numpy.log10(
    2300.0), 100000, dtype=numpy.float64)

# using ExoMol CO (12C-16O). HITRAN and HITEMP are also supported.
autoxs = AutoXS(nuslog, 'ExoMol', 'CO', xsmode='MODIT')

logt=True
T0=1215.0
T1=900.0

if logt:
    lnT0=np.log(T0)
    lnT1=np.log(T1)
    Tx = np.exp((lnT0+lnT1)/2.0)
else:
    Tx = (T0+T1)/2.0


xsv0 = autoxs.xsection(T0, 1.0)  # cross section for 1000K, 1bar (cm2)
xsv1 = autoxs.xsection(T1, 1.0)  # cross section for 1000K, 1bar (cm2)
xsvx = autoxs.xsection(Tx, 1.0)  # cross section for 1000K, 1bar (cm2)
xsvinterp = (xsv0+xsv1)/2.0
fig = plt.figure()
ax = fig.add_subplot(211)
#plt.plot(nus,xsv0)
#plt.plot(nus,xsv1)
plt.title("T0="+str(T0)+"K T1="+str(T1)+"K then interpolation at T="+str(round(Tx,1))+" K")
plt.plot(nuslog,xsvx,label="direct")
plt.plot(nuslog,xsvinterp,label="simple interp")
plt.yscale("log")
plt.xlim(1905,1922)
plt.legend()
ax = fig.add_subplot(212)
plt.plot(nuslog,1.0 - xsvinterp/xsvx,label="diff")
plt.xlim(1905,1922)
plt.legend()
plt.savefig("interpgrid.png")
plt.show()
