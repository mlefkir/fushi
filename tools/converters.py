
def nu2xmm(x):
    """
    Convert a NuSTAR time to an XMM-Newton time
    
    x: NuSTAR time in seconds
    """
    
    mjd_xmm = 50814.0  # xmm.header["MJDREF"]
    # nustar.header["MJDREFI"]+nustar.header["MJDREFF"]
    mjd_nustar = 55197 + 0.00076601852
    return mjd_nustar*86400 + x - mjd_xmm*86400


def xmm2nu(x):
    """
    Convert an XMM-Newton time to a NuSTAR time
    
    x: XMM-Newton time in seconds
    """
    
    mjd_xmm = 50814.0  # hdu_xmm[1].header["MJDREF"]
    # nustar.header["MJDREFI"]+nustar.header["MJDREFF"]
    mjd_nustar = 55197 + 0.00076601852
    return x + mjd_xmm*86400 - mjd_nustar*86400
