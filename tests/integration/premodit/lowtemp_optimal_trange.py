from jax.config import config

config.update("jax_enable_x64", True)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def test_low_temp_trange():
    import numpy as np
    from exojax.spec.api import MdbHitemp
    nu_start = 11353.636363636364
    nu_end = 11774.70588235294
    mdb = MdbHitemp("CH4", nurange=[nu_start,nu_end], elower_max=4000.0)
    from exojax.spec.opacalc import OpaPremodit
    from exojax.utils.grids import wavenumber_grid
    N = 10000
    nus, wav, res = wavenumber_grid(nu_start, nu_end, N, xsmode="premodit") 
    opa = OpaPremodit(mdb, nu_grid=nus, allow_32bit=True, auto_trange=[80.0, 400.0], diffmode=0)

if __name__ == "__main__":
    test_low_temp_trange()