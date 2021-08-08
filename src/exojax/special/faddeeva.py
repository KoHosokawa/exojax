"""special functions in exojax

   exojax.scipy.special  -- special functions

"""

from jax import jit
from jax import custom_vjp
import jax.numpy as jnp
from jax.lax import scan
from exojax.special.erfcx import erfcx

an=jnp.array([ 0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. ,10.5, 11. , 11.5, 12. , 12.5, 13. , 13.5])

a2n2=jnp.array([  0.25,   1.  ,   2.25,   4.  ,   6.25,   9.  ,  12.25, 16.  ,  20.25,  25.  ,  30.25,  36.  ,  42.25,  49.  ,56.25,  64.  ,  72.25,  81.  ,  90.25, 100.  , 110.25, 121.  , 132.25, 144.  , 156.25, 169.  , 182.25])

#expma2n2=jnp.array([7.78800786e-01, 3.67879450e-01, 1.05399221e-01,1.83156393e-02, 1.93045416e-03, 1.23409802e-04,4.78511765e-06, 1.12535176e-07, 1.60522795e-09,1.38879429e-11, 7.28772443e-14, 2.31952270e-16,4.47773235e-19, 5.24288570e-22, 3.72336308e-25,1.60381082e-28, 4.19009311e-32, 6.63967696e-36,6.38149919e-40, 3.78350585e-44, 0.00000000e+00,0.00000000e+00, 0.00000000e+00, 0.00000000e+00,0.00000000e+00, 0.00000000e+00, 0.00000000e+00])

expma2n2_=jnp.array([7.78800786e-01, 3.67879450e-01, 1.05399221e-01,1.83156393e-02, 1.93045416e-03, 1.23409802e-04,4.78511765e-06, 1.12535176e-07])

a2n2_=jnp.array([  0.25,   1.  ,   2.25,   4.  ,   6.25,   9.  ,  12.25, 16. ])


@jit
def rewofz(x,y):
    """Real part of wofz function based on Algorithm 916
    
    We apply a=0.5 for Algorithm 916.
    
    Args:
        x: x < ncut/2 
        y:
        
    Returns:
         jnp.array: Real(wofz(x+iy))

    """
    xy=x*y
    xyp=xy/jnp.pi
    exx=jnp.exp(-x*x)
    f=exx*erfcx(y)*jnp.cos(2.0*xy)+x*jnp.sin(xy)/jnp.pi*exx*jnp.sinc(xyp)

    vec23=jnp.exp(-(an+x)**2)+jnp.exp(-(an-x)**2)
    Sigma23=jnp.sum(vec23/(a2n2+y*y))
    vecbase=exx*expma2n2_/(a2n2_+y*y)
    Sigma1=jnp.sum(vecbase)
    f = f + y/jnp.pi*(-jnp.cos(2.0*xy)*Sigma1 + 0.5*Sigma23)
    return f




@jit
def imwofz(x,y):
    """Imaginary part of wofz function based on Algorithm 916
    
    We apply a=0.5 for Algorithm 916.
    
    Args:
        x: x < ncut/2 
        y:
        
    Returns:
         jnp.array: Imag(wofz(x+iy))

    """
    xy=x*y                             
    xyp=2.0*xy/jnp.pi                      
    exx=jnp.exp(-x*x)                  
    f=-exx*erfcx(y)*jnp.sin(2.0*xy)+x/jnp.pi*exx*jnp.sinc(xyp)           
    vec0=1.0/(a2n2+ y*y)
    vec1=jnp.exp(-(a2n2+x*x))   
    Sigma1=jnp.dot(vec0,vec1)
    vecm=an*vec0
    vec4=jnp.exp(-(an+x)*(an+x)) 
    vec5=jnp.exp(-(an-x)*(an-x))
    
    Sigma45=jnp.dot(vecm,vec5-vec4)
    f = f + 1.0/jnp.pi*(y*jnp.sin(2.0*xy)*Sigma1 + 0.5*Sigma45)

    return f


@jit
def wofzs2(x,y):
    """Asymptotic representation of wofz function 1 for |z|**2 > 112 (for e = 10e-6)

    See Zaghloul (2018) arxiv:1806.01656
    
    Args:
        x: 
        y:
        
    Returns:
         jnp.array, jnp.array: H=real(wofz(x+iy)),L=imag(wofz(x+iy))

    """

    z=x+y*(1j)
    a=1.0/(2.0*z*z)
    q=(1j)/(z*jnp.sqrt(jnp.pi))*(1.0 + a*(1.0 + a*(3.0 + a*15.0)))
    return jnp.real(q),jnp.imag(q)

@jit
def rewofzs2(x,y):
    """Real part of Asymptotic representation of wofz function 1 for |z|**2 > 112 (for e = 10e-6)

    See Zaghloul (2018) arxiv:1806.01656
    
    Args:
        x: 
        y:
        
    Returns:
         jnp.array: real(wofz(x+iy))

    """

    z=x+y*(1j)
    a=1.0/(2.0*z*z)
    q=(1j)/(z*jnp.sqrt(jnp.pi))*(1.0 + a*(1.0 + a*(3.0 + a*15.0)))
    return jnp.real(q)


@jit
def imwofzs2(x,y):
    """Imag part of Asymptotic representation of wofz function 1 for |z|**2 > 112 (for e = 10e-6)

    See Zaghloul (2018) arxiv:1806.01656
    
    Args:
        x: 
        y:
        
    Returns:
         jnp.array: imag(wofz(x+iy))

    """

    z=x+y*(1j)
    a=1.0/(2.0*z*z)
    q=(1j)/(z*jnp.sqrt(jnp.pi))*(1.0 + a*(1.0 + a*(3.0 + a*15.0)))
    return jnp.imag(q)



@custom_vjp
def rewofzx(x, y):
    """[VJP custom defined] Real part of wofz function based on Algorithm 916
    
    We apply a=0.5 for Algorithm 916.
    
    Args:
        x: x < ncut/2 
        y:
        
    Returns:
        jnp.array: Real(wofz(x+iy))

    """
    xy=x*y
    xyp=xy/jnp.pi
    exx=jnp.exp(-x*x)
    f=exx*erfcx(y)*jnp.cos(2.0*xy)+x*jnp.sin(xy)/jnp.pi*exx*jnp.sinc(xyp)

    vec23=jnp.exp(-(an+x)**2)+jnp.exp(-(an-x)**2)
    Sigma23=jnp.sum(vec23/(a2n2+y*y))
    vecbase=exx*expma2n2_/(a2n2_+y*y)
    Sigma1=jnp.sum(vecbase)
    f = f + y/jnp.pi*(-jnp.cos(2.0*xy)*Sigma1 + 0.5*Sigma23)
    return f

def h_fwd(x, y):
    hh=rewofzx(x, y)
    return hh, (hh, imwofz(x, y), x, y)

def h_bwd(res, g):
    """backward

    Note: 
        V=Real(wofz), L=Imag(wofz)

    Args:
        res: res  from h_fwd
        g: g

    Returns:
           jnp.array, jnp.array: g* partial_x h(x,y), g* partial_y h(x,y)


    """
    V, L, x, y = res 
    return (2.0 * (y*L - x*V) * g , 2.0 * (x*L + y*V) * g - 2.0/jnp.sqrt(jnp.pi))

rewofzx.defvjp(h_fwd, h_bwd)
