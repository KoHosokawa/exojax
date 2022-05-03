"""Pseudo line generator

"""
import numpy as np
from exojax.spec.dit import npgetix
import tqdm
import jax.numpy as jnp
import warnings
from exojax.utils.constants import hcperk
from exojax.spec.hitran import SijT


def plg_elower_addcon(indexa,Na,cnu,indexnu,nu_grid,mdb,elower_grid=None,Nelower=10,Ncrit=0,reshape=False, weedout=False, Tpred=296., preov=0.):
    """Pseudo Line Grid for elower w/ an additional condition
    
    Args:
       indexa: the indexing of the additional condition
       Na: the number of the additional condition grid
       cnu: contribution of wavenumber for LSD
       indexnu: nu index
       nugrid: nu grid
       mdb: molecular database (instance made by the MdbExomol/MdbHit class in moldb.py)
       elower_grid: elower_grid (optional)
       Nelower: # of division of elower between min to max values (when elower_grid is not given)
       Ncrit: frrozen line number per bin
       reshape: reshaping output arrays
       weedout: Is it ok to remove weakest lines or not?
       Tpred: typical temperature in the atmosphere
       preov: ad hoc parameter to prevent overflow

    Returns:
       qlogsij0: pseudo logsij0
       qcnu: pseudo cnu
       num_unique: number of lines in grids
       elower_grid: elower of pl
       frozen_mask: mask for frozen lines into pseudo lines 
       nonzeropl_mask: mask for pseudo-lines w/ non-zero

    """
    elower = mdb.elower
    Nnugrid=len(nu_grid)
    Tref = 296.0
    Tpred = Tpred * 1.0
    preov = max(- hcperk*(elower/Tpred - elower/Tref)) - 80. if preov==0. else preov
    #kT0=10000.0
    warnings.simplefilter('error')
    try:
        expme = np.exp(- hcperk*(elower/Tpred - elower/Tref) - preov)
    except RuntimeWarning as e:
        raise Exception(str(e)+' :\t Please adjust "preov"...')
    if elower_grid is None:
        margin = 1.0
        min_expme = np.exp(- hcperk*((min(elower)-margin)/Tpred - (min(elower)-margin)/Tref) - preov)
        max_expme = np.exp(- hcperk*((max(elower)+margin)/Tpred - (max(elower)+margin)/Tref) - preov)
        expme_grid = np.linspace(min_expme, max_expme, Nelower)
        elower_grid = (np.log(expme_grid) + preov) / (-hcperk) / (1/Tpred - 1/Tref)
    else:
        expme_grid=np.exp(-elower_grid/kT0)
        Nelower=len(expme_grid)
    warnings.simplefilter('default')

    qlogsij0,qcnu,num_unique,frozen_mask=get_qlogsij0_addcon(indexa,Na,cnu,indexnu,Nnugrid,mdb,expme,expme_grid,Ncrit=Ncrit, Tpred=Tpred) #, elower_grid=elower_grid, Nelower=10
    
    nonzeropl_mask=qlogsij0>-np.inf
    '''if weedout:
        qlogsij0_tr = np.log(np.exp(qlogsij0))
        nonzeropl_mask=(qlogsij0_tr>-np.inf) & (qlogsij0_tr<0)
    else:
        nonzeropl_mask = qlogsij0<0'''
    
    Nline=len(elower)
    Nunf=np.sum(~frozen_mask)
    Npl=len(qlogsij0[nonzeropl_mask])
    print("# of original lines:",Nline)        
    print("# of unfrozen lines:",Nunf)
    print("# of frozen lines:",np.sum(frozen_mask))
    print("# of pseudo lines:",Npl)
    print("# of total lines:",(Npl+Nunf))
    print("# compression:",(Npl+Nunf)/Nline)

    if reshape==True:
        qlogsij0=qlogsij0.reshape(Na,Nnugrid,Nelower)
        qcnu=qcnu.reshape(Na,Nnugrid,Nelower)
        num_unique=num_unique.reshape(Na,Nnugrid,Nelower)
        
    return qlogsij0,qcnu,num_unique,elower_grid,frozen_mask,nonzeropl_mask

def get_qlogsij0_addcon(indexa,Na,cnu,indexnu,Nnugrid,mdb,expme,expme_grid, Ncrit=0, Tpred=296.):
    """gether (freeze) lines w/ additional indexing

    Args:
       indexa: the indexing of the additional condition
       Na: the number of the additional condition grid
       cnu: contribution of wavenumber for LSD
       indexnu: index of wavenumber
       Nnugrid: number of nu grid
       mdb: molecular database (instance made by the MdbExomol/MdbHit class in moldb.py)
       expme: exp(-elower/kT0)
       expme_grid: exp(-elower/kT0)_grid
       Ncrit:
       Tpred:

    """
    m=len(expme_grid)
    n=m*Nnugrid
    Ng=n*Na
    
    cont,index=npgetix(expme,expme_grid) #elower contribution and elower index of lines
    eindex=index+m*indexnu+n*indexa #extended index elower,nu,a
    
    #frozen criterion
    num_unique=np.bincount(eindex,minlength=Ng) # number of the lines in a bin
    
    lmask=(num_unique>=Ncrit)
    erange=range(0,Ng)
    frozen_eindex=np.array(erange)[lmask]
    frozen_mask=np.isin(eindex,frozen_eindex)
    
    SijTpred_frozen = SijT(Tpred, \
                    mdb.logsij0[frozen_mask], mdb.nu_lines[frozen_mask], mdb.elower[frozen_mask], \
                    qT=mdb.qr_interp(Tpred))
    persist_freezing = SijTpred_frozen < max(SijTpred_frozen)/1000
    index_persist_freezing = np.where(frozen_mask)[0][persist_freezing]
    frozen_mask = np.isin(np.arange(len(frozen_mask)), index_persist_freezing)
        
    Sij=np.exp(mdb.logsij0)
    #qlogsij0
    qlogsij0=np.bincount(eindex,weights=Sij*(1.0-cont)*frozen_mask,minlength=Ng)
    qlogsij0=qlogsij0+np.bincount(eindex+1,weights=Sij*cont*frozen_mask,minlength=Ng)
    qlogsij0=np.log(qlogsij0)

    '''#qlogsij0
    Tref = 296.0
    qlogsij0 = np.bincount(eindex, weights = \
                      (jnp.log(jnp.exp(logsij0 - hcperk*(elower/Tpred - elower/Tref)) * (1. - cont)) + \
                       hcperk*(elower_grid[index]/Tpred - elower_grid[index]/Tref) )*frozen_mask, minlength=Ng)
    qlogsij0 = qlogsij0 + np.bincount(eindex+1, weights = \
                      (jnp.log(jnp.exp(logsij0 - hcperk*(elower/Tpred - elower/Tref)) * (cont)) + \
                                  hcperk*(elower_grid[index+1]/Tpred - elower_grid[index+1]/Tref) )*frozen_mask, minlength=Ng)'''

    #qcnu
    qcnu_den=np.bincount(eindex,weights=Sij*frozen_mask,minlength=Ng)
    qcnu_den=qcnu_den+np.bincount(eindex+1,weights=Sij*frozen_mask,minlength=Ng)
    qcnu_den[qcnu_den==0.0]=1.0
    
    qcnu_num=np.bincount(eindex,weights=Sij*cnu*frozen_mask,minlength=Ng)
    qcnu_num=qcnu_num+np.bincount(eindex+1,weights=Sij*cnu*frozen_mask,minlength=Ng)
    qcnu=qcnu_num/qcnu_den

    return qlogsij0,qcnu,num_unique,frozen_mask

def plg_elower(cnu,indexnu,Nnugrid,logsij0,elower,elower_grid=None,Nelower=10,Ncrit=0,reshape=True):
    """Pseudo Line Grid for elower
    
    Args:
       cnu: contribution of wavenumber for LSD
       indexnu: index of wavenumber
       Nnugrid: number of Ngrid
       logsij0: log line strength
       elower: elower
       elower_grid: elower_grid (optional)
       Nelower: # of division of elower between min to max values when elower_grid is not given
       Ncrit: frrozen line number per bin

    Returns:
       qlogsij0
       qcnu
       num_unique
       elower_grid

    """
    
    kT0=10000.0
    expme=np.exp(-elower/kT0)
    if elower_grid is None:
        margin=1.0
        min_expme=np.min(expme)*np.exp(-margin/kT0)
        max_expme=np.max(expme)*np.exp(margin/kT0)
        expme_grid=np.linspace(min_expme,max_expme,Nelower)
        elower_grid=-np.log(expme_grid)*kT0
    else:
        expme_grid=np.exp(-elower_grid/kT0)
        Nelower=len(expme_grid)
        
    qlogsij0,qcnu,num_unique,frozen_mask=get_qlogsij0(cnu,indexnu,Nnugrid,logsij0,expme,expme_grid,Nelower=10,Ncrit=Ncrit)

    if reshape==True:
        qlogsij0=qlogsij0.reshape(Nnugrid,Nelower)
        qcnu=qcnu.reshape(Nnugrid,Nelower)
        num_unique=num_unique.reshape(Nnugrid,Nelower)
    
    print("# of unfrozen lines:",np.sum(~frozen_mask))
    print("# of pseudo lines:",len(qlogsij0[qlogsij0>0.0]))
    
    return qlogsij0,qcnu,num_unique,elower_grid

def get_qlogsij0(cnu,indexnu,Nnugrid,logsij0,expme,expme_grid,Nelower=10,Ncrit=0):
    """gether (freeze) lines 

    Args:
       cnu: contribution of wavenumber for LSD
       indexnu: index of wavenumber
       logsij0: log line strength
       expme: exp(-elower/kT0)
       expme_grid: exp(-elower/kT0)_grid
       Nelower: # of division of elower between min to max values

    """
    m=len(expme_grid)
    cont,index=npgetix(expme,expme_grid) #elower contribution and elower index of lines
    eindex=index+m*indexnu #extended index
    
    #frozen criterion
    Ng=m*Nnugrid
    num_unique=np.bincount(eindex,minlength=Ng) # number of the lines in a bin
    
    lmask=(num_unique>=Ncrit)
    erange=range(0,Ng)
    frozen_eindex=np.array(erange)[lmask]
    frozen_mask=np.isin(eindex,frozen_eindex)
        
    Sij=np.exp(logsij0)
    #qlogsij0
    qlogsij0=np.bincount(eindex,weights=Sij*(1.0-cont)*frozen_mask,minlength=Ng)
    qlogsij0=qlogsij0+np.bincount(eindex+1,weights=Sij*cont*frozen_mask,minlength=Ng)    
    qlogsij0=np.log(qlogsij0)

    #qcnu
    qcnu_den=np.bincount(eindex,weights=Sij*frozen_mask,minlength=Ng)
    qcnu_den=qcnu_den+np.bincount(eindex+1,weights=Sij*frozen_mask,minlength=Ng)
    qcnu_den[qcnu_den==0.0]=1.0
    
    qcnu_num=np.bincount(eindex,weights=Sij*cnu*frozen_mask,minlength=Ng)
    qcnu_num=qcnu_num+np.bincount(eindex+1,weights=Sij*cnu*frozen_mask,minlength=Ng)
    qcnu=qcnu_num/qcnu_den

    return qlogsij0,qcnu,num_unique,frozen_mask


def gather_lines(mdb,Na,Nnugrid,Nelower,nu_grid,cnu,indexnu,qlogsij0,qcnu,elower_grid,alpha_ref_grid,n_Texp_grid,frozen_mask,nonzeropl_mask):
    """gather pseudo lines and unfrozen lines into lines for exomol

    Args:

       mdb: molecular database
       Na: the number of the additional condition grid (gammaL set for Exomol)
       Nnugrid: # of nu_grid
       Nelower: # of elower grid
       nu_grid: nu grid
       cnu: contribution of wavenumber for LSD
       indexnu: index nu
       qlogsij0: log line strength
       qcnu: pseudo line, contribution of wavenumber for LSD
       elower_grid: elower_grid
       alpha_ref_grid: grid of alpha_ref
       n_Texp_grid: grid of n_Texp
       frozen_mask: mask for frozen lines into pseudo lines 
       nonzeropl_mask: mask for pseudo-lines w/ non-zero

    Returns:
       mdb for the gathered lines
       cnu for the gathered lines
       indexnu for the gathered lines


    """
    
    
    #gathering
    ## q-part should be ((Na,Nnugrid,Nelower).flatten)[nonzeropl_mask]
    import jax.numpy as jnp

    ## MODIT
    arrone=np.ones((Na,Nelower))    
    qnu_grid=(arrone[:,np.newaxis,:]*nu_grid[np.newaxis,:,np.newaxis]).flatten()
    indexnu_grid=np.array(range(0,len(nu_grid)),dtype=int)
    qindexnu=(arrone[:,np.newaxis,:]*indexnu_grid[np.newaxis,:,np.newaxis]).flatten()
    cnu=np.hstack([qcnu[nonzeropl_mask],cnu[~frozen_mask]])
    indexnu=np.array(np.hstack([qindexnu[nonzeropl_mask],indexnu[~frozen_mask]]),dtype=int)
    
    #mdb
    mdb.logsij0=np.hstack([qlogsij0[nonzeropl_mask],mdb.logsij0[~frozen_mask]])
    mdb.nu_lines=np.hstack([qnu_grid[nonzeropl_mask],mdb.nu_lines[~frozen_mask]])
    mdb.dev_nu_lines=jnp.array(mdb.nu_lines)

    onearr=np.ones((Na,Nnugrid))
    qelower=(onearr[:,:,np.newaxis]*elower_grid[np.newaxis,np.newaxis,:]).flatten()
    mdb.elower=np.hstack([qelower[nonzeropl_mask],mdb.elower[~frozen_mask]])
    
    #gamma     #Na,Nnugrid,Nelower
    onearr_=np.ones((Nnugrid,Nelower))
    alpha_ref_grid=alpha_ref_grid[:,np.newaxis,np.newaxis]*onearr_
    alpha_ref_grid=alpha_ref_grid.flatten()
    n_Texp_grid=n_Texp_grid[:,np.newaxis,np.newaxis]*onearr_
    n_Texp_grid=n_Texp_grid.flatten()    
    mdb.alpha_ref=np.hstack([alpha_ref_grid[nonzeropl_mask],mdb.alpha_ref[~frozen_mask]])
    mdb.n_Texp=np.hstack([n_Texp_grid[nonzeropl_mask],mdb.n_Texp[~frozen_mask]])
    mdb.A=jnp.zeros_like(mdb.logsij0) #no natural width

    lenarr=[len(mdb.logsij0),len(mdb.elower),len(cnu),len(indexnu),len(mdb.nu_lines),len(mdb.dev_nu_lines),len(mdb.alpha_ref),len(mdb.n_Texp),len(mdb.A)]
    
    Ngat=np.unique(lenarr)
    if len(Ngat)>1:
        print("Error: Length mismatch")
    else:
        print("Nline gathered=",Ngat[0])
    
    return mdb, cnu, indexnu

    
