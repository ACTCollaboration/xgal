from xgal import deltag
import healpy as hp
import numpy as np
from orphics import io

def test_correlated_alm():
    lmax = 2000
    ells = np.arange(0,lmax,1)

    def get_cls(ells,index,amplitude):
        cls = amplitude * ells.astype(np.float32)**index
        cls[ells<2] = 0
        return cls

    Clf1f1 = get_cls(ells,-1,1)
    Clf2f2 = get_cls(ells,-1.3,2)
    Clf1f2 = get_cls(ells,-1.4,0.5)

    alm_f1 = hp.synalm(Clf1f1,lmax=lmax-1)
    alm_f2 = deltag.generate_correlated_alm(alm_f1,Clf1f1,Clf2f2,Clf1f2)

    f1f1 = hp.alm2cl(alm_f1,alm_f1)
    f2f2 = hp.alm2cl(alm_f2,alm_f2)
    f1f2 = hp.alm2cl(alm_f1,alm_f2)

    pl = io.Plotter(xyscale='linlog',scalefn = lambda x: x)
    pl.add(ells,f1f1,color="C0",alpha=0.4)
    pl.add(ells,f2f2,color="C1",alpha=0.4)
    pl.add(ells,f1f2,color="C2",alpha=0.4)
    pl.add(ells,Clf1f1,label="f1f1",color="C0",ls="--",lw=3)
    pl.add(ells,Clf2f2,label="f2f2",color="C1",ls="--",lw=3)
    pl.add(ells,Clf1f2,label="f1f2",color="C2",ls="--",lw=3)
    pl.done()
