"""generate test miegrid

"""
import pkg_resources
from exojax.spec import pardb


def gendata_miegrid():
    """
    generates miegrid for test.refrind
    """
    refrind_path = pkg_resources.resource_filename(
        "exojax", "data/testdata/test.refrind"
    )

    pdb_nh3 = pardb.PdbCloud("test", download=False, refrind_path=refrind_path)
    if True:
        pdb_nh3.generate_miegrid(
            sigmagmin=-1.0,
            sigmagmax=1.0,
            Nsigmag=4,
            rg_max=-4.0,
            Nrg=4,
        )


if __name__ == "__main__":
    gendata_miegrid()
