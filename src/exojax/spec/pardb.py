""" Particulates Database

- Cloud
- Haze (in future)

"""

import numpy as np
import pathlib
from exojax.spec.mie import evaluate_miegrid_layers


__all__ = ["PdbCloud"]


class PdbCloud(object):
    def __init__(
        self,
        condensate,
        nurange=[-np.inf, np.inf],
        margin=10.0,
        path="./.database/particulates/virga",
        gen_miegrid=False,
    ):
        """Particulates Database for clouds

        Args:
            condensate: condensate, such as NH3, H2O, MgSiO3 etc
            nurange: wavenumber range list (cm-1) or wavenumber array
            margin: margin for nurange (cm-1)
            path: database path

        """
        self.path = pathlib.Path(path)
        self.condensate = condensate
        self.download_and_unzip()
        self.load_virga()

        self.nurange = [np.min(nurange), np.max(nurange)]
        self.margin = margin
        self.set_saturation_pressure_list()
        self.set_condensate_density()

        # Mie scattering
        self.ready_mie = False
        self.set_miegrid_filename()
        self.set_miegrid_path()

    def download_and_unzip(self):
        """Downloading virga refractive index data

        Note:
            The download URL is written in exojax.utils.url.
        """
        import urllib.request
        import os
        import shutil
        from exojax.utils.url import url_virga
        from exojax.utils.files import find_files_by_extension
        from exojax.utils.files import get_file_names_without_extension

        try:
            os.makedirs(str(self.path), exist_ok=True)
            filepath = self.path / "virga.zip"
            if (filepath).exists():
                print(
                    str(filepath),
                    " exists. Remove it if you wanna re-download and unzip.",
                )
            else:
                print("Downloading ", url_virga())
                # urllib.request.urlretrieve(url_virga(), str(filepath))
                data = urllib.request.urlopen(url_virga()).read()
                with open(str(filepath), mode="wb") as f:
                    f.write(data)
                shutil.unpack_archive(str(filepath), str(self.path))
            self.virga_condensates = get_file_names_without_extension(
                find_files_by_extension(str(self.path), ".refrind")
            )
            if self.condensate in self.virga_condensates:
                self.refrind_path = self.path / pathlib.Path(
                    self.condensate + ".refrind"
                )
                print("Refractive index file found: ", self.refrind_path)
            else:
                print(
                    "No refrind file found. Refractive indices of ",
                    self.virga_condensates,
                    "are available.",
                )
        except:
            print("VIRGA refractive index download failed")

    def load_virga(self):
        from exojax.spec.unitconvert import wav2nu

        _, wave, nn, kk = np.loadtxt(
            open(self.refrind_path, "rt").readlines(), unpack=True, usecols=[0, 1, 2, 3]
        )

        self.refraction_index_wavenumber = wav2nu(wave, "um")  # wave in micron
        self.refraction_index_wavelength_nm = wave * 1.0e3
        self.refraction_index = nn + kk * (1j)

    def set_saturation_pressure_list(self):
        from exojax.atm.psat import (
            psat_ammonia_AM01,
            psat_water_AM01,
            psat_Fe_AM01,
            psat_enstatite_AM01,
        )

        self.saturation_pressure_solid_list = {
            "NH3": psat_ammonia_AM01,
            "H2O": psat_water_AM01,
            "MgSiO3": psat_enstatite_AM01,
            "Fe": psat_Fe_AM01,
        }

    def set_condensate_density(self):
        from exojax.atm.condensate import condensate_density

        self.rhoc = condensate_density[self.condensate]

    def saturation_pressure(self, temperatures):
        return self.saturation_pressure_solid_list[self.condensate](temperatures)

    def set_miegrid_filename(self, miegrid_filename=None):
        if miegrid_filename is None:
            self.miegrid_filename = "miegrid_lognorm_" + self.condensate + ".mgd"
        elif miegrid_filename == "auto":
            raise ValueError("not implemented yet")
        else:
            self.miegrid_filename = miegrid_filename

    def set_miegrid_path(self, miegrid_path=None):
        if miegrid_path is None:
            self.miegrid_path = self.path / pathlib.Path(self.miegrid_filename + ".npz")
        else:
            self.miegrid_path = pathlib.Path(miegrid_path + ".npz")

        if self.miegrid_path.exists():
            print("Miegrid file exists:", str(self.miegrid_path))
        else:
            print("Miegrid file does not exist at ", str(self.miegrid_path))
            print(
                "Generate miegrid file using pdb.generate_miegrid if you use Mie scattering"
            )

    def load_miegrid(self):
        from exojax.spec.mie import read_miegrid

        if self.miegrid_path.exists():
            self.miegrid, self.rg_arr, self.sigmag_arr = read_miegrid(self.miegrid_path)
            self.ready_mie = True
            print(
                "pdb.miegrid, pdb.rg_arr, pdb.sigmag_arr are now available. The Mie scattering computation is ready."
            )
        else:
            raise ValueError("Miegrid file Not Found.")

    def generate_miegrid(
        self,
        sigmagmin=-1.0,
        sigmagmax=1.0,
        Nsigmag=10,
        rg_min=-7.0,
        rg_max=-3.0,
        Nrg=40,
    ):
        """generates miegrid assuming lognormal size distribution

        Args:
            log_sigmagmin (float, optional): log sigma_g minimum. Defaults to -1.0.
            log_sigmagmax (float, optional): log sigma_g maximum. Defaults to 1.0.
            Nsigmag (int, optional): the number of the sigmag grid. Defaults to 10.
            log_rg_min (float, optional): log r_g (cm) minimum . Defaults to -7.0.
            log_rg_max (float, optional): log r_g (cm) minimum. Defaults to -3.0.
            Nrg (int, optional): the number of the rg grid. Defaults to 40.

        Note:
            it will take a bit long time.

        """

        from exojax.spec.mie import make_miegrid_lognormal

        make_miegrid_lognormal(
            self.pdb,
            str(self.miegrid_filename),
            sigmagmin,
            sigmagmax,
            Nsigmag,
            rg_min,
            rg_max,
            Nrg,
        )
        print(str(self.miegrid_filename), " was generated.")

    def mie_parameters(self, rg_layer, sigmag_layer):
        return evaluate_miegrid_layers(
            rg_layer, sigmag_layer, self.miegrid, self.rg_arr, self.sigmag_arr
        )


if __name__ == "__main__":
    pdb = PdbCloud("NH3")
    pdb.load_miegrid()
