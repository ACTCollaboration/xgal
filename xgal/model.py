"""
xgal.model

Contains functions and classes for converting parameters
to a prediction for the gk and gg data vectors.

"""

def cl(spec,params,zs=None,dndz=None,backend="camb"):
    assert backend in ['orphics','hmvec','camb','ccl']
    assert spec in ['kk','kg','gg']

    if backend=='orphics':
        from orphics import cosmology
        lc = cosmology.LimberCosmology(params,skipCls=True)
        
        
    elif backend=='hmvec':
        pass
    elif backend=='camb':
        pass
    elif backend=='ccl':
        pass
