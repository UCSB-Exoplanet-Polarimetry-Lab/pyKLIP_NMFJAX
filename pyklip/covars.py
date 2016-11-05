import numpy as np
# cython imports
try:
    import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})
    import pyklip.covars_c as covars_c
except ImportError:
    covars_c = None


def _matern32(x, y, sigmas, corr_len):
    """
    Generates a Matern (nu=3/2) covariance matrix that assumes x/y has the same correlation length

    C_ij = \sigma_i \sigma_j (1 + sqrt(3) r_ij / l) exp(-sqrt(3) r_ij / l)

    Args:
        x (np.array): 1-D array of x coordinates
        y (np.array): 1-D array of y coordinates
        sigmas (np.array): 1-D array of errors on each pixel
        corr_len (float): correlation length of the Matern function

    Returns:
        cov (np.array): 2-D covariance matrix parameterized by the Matern function
    """
    r = np.sqrt((x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2)
    arg = np.sqrt(3) * r / corr_len
    cov = sigmas[:, None] * sigmas[None, :] * (1+arg) * np.exp(-arg)
    return cov


def matern32(x, y, sigmas, corr_len, mode=None):
    """
    Generates a Matern (nu=3/2) covariance matrix that assumes x/y has the same correlation length

    C_ij = \sigma_i \sigma_j (1 + sqrt(3) r_ij / l) exp(-sqrt(3) r_ij / l)

    Args:
        x (np.array): 1-D array of x coordinates
        y (np.array): 1-D array of y coordinates
        sigmas (np.array): 1-D array of errors on each pixel
        corr_len (float): correlation length of the Matern function
        mode (string): either "numpy", "cython", or None, specifying the implementation of the kernel.
                       if None, it will pick the default (currently, numpy)

    Returns:
        cov (np.array): 2-D covariance matrix parameterized by the Matern function
    """
    # error check to see which implementation of the kernel to use
    if mode is None:
        mode = "numpy"
    elif mode == "cython" and covars_c is None:
        raise ValueError("cython code covars_c.pyx cannot be compiled. Check your cython installation.")


    if mode == "cython":
        return covars_c.matern32(x, y, sigmas, corr_len)
    elif mode == "numpy":
        return _matern32(x, y, sigmas, corr_len)
    else:
        raise ValueError("mode must either be 'cython' or 'numpy'")


def _sq_exp(x, y, sigmas, corr_len):
    """
    Generates square exponential covariance matrix that assumes x/y has the same correlation length

    C_ij = \sigma_i \sigma_j exp(-r_ij^2/[2 l^2])

    Args:
        x (np.array): 1-D array of x coordinates
        y (np.array): 1-D array of y coordinates
        sigmas (np.array): 1-D array of errors on each pixel
        corr_len (float): correlation length (i.e. standard deviation of Gaussian)

    Returns:
        cov (np.array): 2-D covariance matrix parameterized by the Matern function
    """
    r = np.sqrt((x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2)
    arg = r**2 / (2 * corr_len**2)
    cov = sigmas[:, None] * sigmas[None, :] * np.exp(-arg)
    return cov


def sq_exp(x, y, sigmas, corr_len, mode=None):
    """
    Generates square exponential covariance matrix that assumes x/y has the same correlation length

    C_ij = \sigma_i \sigma_j exp(-r_ij^2/[2 l^2])

    Args:
        x (np.array): 1-D array of x coordinates
        y (np.array): 1-D array of y coordinates
        sigmas (np.array): 1-D array of errors on each pixel
        corr_len (float): correlation length (i.e. standard deviation of Gaussian)
        mode (string): either "numpy", "cython", or None, specifying the implementation of the kernel.
                       if None, it will pick the default (currently, numpy)

    Returns:
        cov (np.array): 2-D covariance matrix parameterized by the Matern function
    """
    # error check to see which implementation of the kernel to use
    if mode is None:
        mode = "numpy"
    elif mode == "cython" and covars_c is None:
        raise ValueError("cython code covars_c.pyx cannot be compiled. Check your cython installation.")


    if mode == "cython":
        return covars_c.sq_exp(x, y, sigmas, corr_len)
    elif mode == "numpy":
        return _sq_exp(x, y, sigmas, corr_len)
    else:
        raise ValueError("mode must either be 'cython' or 'numpy'")
