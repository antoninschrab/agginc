"""
Numpy implementation
This file contains our three tests:
MMDAggInc, HSICAggInc and KSDAggInc
which are implemented in the function agginc()
For details, see our paper:
Efficient Aggregated Kernel Tests using Incomplete U-statistics
Antonin Schrab, Ilmun Kim, Benjamin Guedj, Arthur Gretton
"""

import numpy as np
import itertools
import scipy.spatial
import psutil


def agginc(
        agg_type,
        X,
        Y,
        R=200,
        alpha=0.05,
        weights_type="uniform",
        number_bandwidths=10,
        hsic_collection_parameters=(2, -2, 2),
        batch_size="auto",
        memory_percentage=0.8,
        B1=500,
        B2=500,
        B3=50,
        seed=42,
        return_dictionary=False,
        bandwidths=None,
    ):
    """
    Efficient Aggregated tests for two-sample (MMD), independence (HSIC) 
    and goodness-of-fit (KSD) testing.
    
    Given the appropriate data for the type of testing, 
    return 0 if the test fails to reject the null (i.e. same distribution, independent, fits the data), 
    or return 1 if the test rejects the null (i.e. different distribution, dependent, does not fit the data).
    
    Parameters
    ----------
    agginc: str
        "mmd" or "hsic" or "ksd"
    X : array_like
        The shape of X must be of the form (N_X, d_X) where N_X is the number
        of samples and d_X is the dimension.
    Y : array_like
        The shape of Y must be of the form (N_Y, d_Y) 
        where N_Y is the number of samples and d_Y is the dimension.
        Case agginc = "mmd": Y is the second sample, we must have d_X = d_Y.
        Case agginc = "hsic": Y is the paired sample, we must have N_X = N_Y.
        Case agginc = "ksd": Y is the score of X, we must have N_Y = N_X and d_Y = d_X.
    R : int
        Number of superdiagonals to consider. 
        If R >= min(N_X, N_Y) - 1 then the complete U-statistic is computed in quadratic time.
    alpha: float
        The level alpha must be between 0 and 1.
    weights_type: str
        Must be "uniform", or "centred", or "increasing", or "decreasing".
    number_bandwidths: int
        The number of bandwidths to include in the collection when agginc is "mmd" or "ksd".
    hsic_collection_parameters : tuple
        Tuple of length either 3 or 5.
        Case tuple of length 3:
            hsic_collection_parameters = (power, l_minus, l_plus)
            collection = (power ** i * median_bandwidth_X, power ** j * median_bandwidth_Y)
                         for i, j = l_minus, ..., l_plus
        Case tuple of length 5:
            hsic_collection_parameters = (power, l_minus_X, l_plus_X, l_minus_Y, l_plus_Y)
            collection = (power ** i * median_bandwidth_X, power ** j * median_bandwidth_Y)
                         for i = l_minus_X, ..., l_plus_X
                         for j = l_minus_Y, ..., l_plus_Y
    batch_size : int or None or str
        The memory cost consists in storing an array of shape (batch_size, R * N - R * (R - 1) / 2)
        where batch_size is between 1 and B1 + B2.
        Using batch_size = "auto", calculates automatically the batch_size which uses 80% of the memory.
        For faster runtimes but using more memory, use batch_size = None (equivalent to batch_size = B1 + B2)
        By decreasing batch_size from B1 + B2, the memory cost is reduced but the runtimes increase.
    memory_percentage: float
        The value of memory_percentage must be between 0 and 1.
        It is used when batch_size = "auto", the batch_size is calculated automatically to use memory_percentage of the memory.
    B1: int
        B1 is the number of wild bootstrap samples to approximate the quantiles.
    B2: int
        B2 is the number of wild bootstrap samples to approximate the level correction.
    B3: int
        Number of steps of bissection method to perform to estimate the level correction.
    seed: int 
        Random seed used for the randomness of the Rademacher variables.
    return_dictionary: bool
        If true, a dictionary is also returned containing the overall test out and for each single test: 
        the test output, the kernel, the bandwidth, the statistic value, the quantile value, 
        the p-value and the p-value threshold value.
   bandwidths: array_like or list or None
        If bandwidths is None, the collection of bandwidths is computed automatically.
        Otherwise, the collection provided in bandwidths is used instead.
        If agg_type is "mmd" or "ksd", then bandwidths needs to be an array of one dimension.
        If agg_type is "hsic", then bandwidths should be a list containing 2 elements:
            first element is a list or array of bandwidths for X,
            second element is a list or array of bandwidths for Y.
        Note that number_bandwidths and hsic_collection_parameters are overwritten by bandwidths.

        
    Returns
    -------
    output : int
        0 if the AggInc test fails to reject the null (i.e. same distribution, independent, fits the data), 
        1 if the AggInc test rejects the null (i.e. different distribution, dependent, does not fit the data).
    dictionary: dict
        Returned only if return_dictionary is True.
        Dictionary containing the overall output of the AggInc test, and for each single test: 
        the test output, the kernel, the bandwidth, the statistic, the quantile, 
        the p-value and the p-value threshold.
    
    
    Examples
    --------
    Check out the notebooks demo.ipynb and speed.ipynb for examples and speed comparisons.
    
    >>> # MMDAggInc
    >>> rs = np.random.RandomState(0)
    >>> X = rs.randn(500, 10)
    >>> Y = rs.randn(500, 10) + 1
    >>> output = agginc("mmd", X, Y)
    >>> output
    1
    >>> output, dictionary = agginc("mmd", X, Y, return_dictionary=True)
    >>> output
    1
    >>> dictionary
    {'MMDAggInc test reject': True,
     'Single test 1': {'Reject': True,
      'Kernel': 'Gaussian',
      'Bandwidth': 0.5615753377127076,
      'MMD': 3.841147128497093e-06,
      'MMD quantile': 2.131735610252617e-06,
      'p-value': 0.001996007984031936,
      'p-value threshold': 0.0259481037924143},
      ...
    }
    
    >>> # HSICAggInc
    >>> rs = np.random.RandomState(0)
    >>> X = rs.randn(500, 10)
    >>> Y = 0.5 * X + rs.randn(500, 10)
    >>> output = agginc("hsic", X, Y)
    >>> output
    1
    >>> output, dictionary = agginc("hsic", X, Y, return_dictionary=True)
    >>> output
    1
    >>> dictionary
    {'HSICAggInc test reject': True,
      'Single test 1': {'Reject': False,
      'Kernel': 'Gaussian',
      'Bandwidth X': 1.0690827149076558,
      'Bandwidth Y': 1.195839319784294,
      'HSIC': 1.118804471773042e-06,
      'HSIC quantile': 2.070241076091771e-06,
      'p-value': 0.1377245508982036,
      'p-value threshold': 0.007984031936127067},
      ...
    }
    
   >>> # KSDAggInc
    >>> perturbation = 0.5
    >>> rs = np.random.RandomState(0)
    >>> X = rs.gamma(5 + perturbation, 5, (500, 1))
    >>> score_gamma = lambda x, k, theta : (k - 1) / x - 1 / theta
    >>> score_X = score_gamma(X, 5, 5)
    >>> output = agginc("ksd", X, score_X)
    >>> output
    1
    >>> output, dictionary = agginc("ksd", X, score_X, return_dictionary=True)
    >>> output
    1
    >>> dictionary
    {'KSDAggInc test reject': True,
      'Single test 1': {'Reject': False,
      'Kernel': 'imq',
      'Bandwidth': 1.0,
      'KSD': 0.0005635488974350794,
      'KSD quantile': 0.0011720189723149017,
      'p-value': 0.13373253493013973,
      'p-value threshold': 0.009980039920159278},
      ...
    }
    """
    # Warnings
    assert memory_percentage > 0 and memory_percentage <= 1
    if number_bandwidths != 10 and agg_type == "hsic":
        warnings.warn("Parameter 'number_bandwidths' is not at its default value. Note that this parameter has no effect on the HSICAggInc test. The collection of bandwidths for HSICAggInc can be varied by changing the parameter 'hsic_collection_parameters'. Alternatively, a custom collection can be provided using the parameter 'bandwidths'.")
    if agg_type != "hsic" and hsic_collection_parameters != (2, -2, 2):
        warnings.warn("Parameter 'hsic_collection_parameters' is not at its default value. Note that this parameter has no effect on the MMDAggInc and KSDAggInc tests. The size of the collection of bandwidths for those tests can be varied by changing the parameter 'number_bandwidths'. Alternatively, a custom collection can be provided using the parameter 'bandwidths'.")
        
    # function compute_h_values
    if agg_type == "mmd":
        compute_h_values = compute_h_MMD_values
    elif agg_type == "hsic":
        compute_h_values = compute_h_HSIC_values
    elif agg_type == "ksd":
        compute_h_values = compute_h_KSD_values
    else:
        raise ValueError('The value of agg_type should be "mmd" or "hsic" or "ksd".')
        
    # collection of bandwidths
    if bandwidths is not None:
        if agg_type == "mmd" or agg_type == "ksd":
            assert bandwidths.ndim == 1
            number_bandwidths = len(bandwidths)
        elif agg_type == "hsic":
            assert len(bandwidths) == 2 and len(bandwidths[0]) > 0 and len(bandwidths[0]) > 0
            number_bandwidths = len(bandwidths[0]) * len(bandwidths[1])
    elif agg_type == "mmd":
        Z = np.concatenate((X, Y))
        max_samples = 500
        distances = scipy.spatial.distance.cdist(X[:max_samples], Y[:max_samples], "euclidean").reshape(-1)
        if np.min(distances) < 10 ** (-1):
            dd = np.sort(distances)
            lambda_min = np.maximum(dd[int(np.floor(len(dd) * 0.05))], 10 ** (-1))
        else:
            lambda_min = np.min(distances)
        lambda_min = lambda_min / 2
        lambda_max = np.maximum(np.max(distances), 3 * 10 ** (-1))
        lambda_max = lambda_max * 2
        power = (lambda_max / lambda_min) ** (1 / (number_bandwidths - 1))
        bandwidths = np.array([power ** i * lambda_min for i in range(number_bandwidths)])
    elif agg_type == "hsic":
        assert weights_type == "uniform"
        if len(hsic_collection_parameters) == 3:
            power, l_minus, l_plus = hsic_collection_parameters
            l_minus = (l_minus, l_minus)
            l_plus = (l_plus, l_plus)
        elif len(hsic_collection_parameters) == 5:
            power, l_minus_X, l_plus_X, l_minus_Y, l_plus_Y = hsic_collection_parameters
            l_minus = (l_minus_X, l_minus_Y)
            l_plus = (l_plus_X, l_plus_Y)
        else:
            raise ValueError(
                "hsic_collection_parameters should have length either 3 or 5."
            )
        max_samples=500
        distances = scipy.spatial.distance.pdist(X[:max_samples], "euclidean") 
        median_bandwidth_X = np.median(distances[distances > 0])
        distances = scipy.spatial.distance.pdist(Y[:max_samples], "euclidean")  
        median_bandwidth_Y = np.median(distances[distances > 0])
        median_bandwidths = (median_bandwidth_X, median_bandwidth_Y)
        bandwidths = [
            [power ** i * median_bandwidths[j] for i in range(l_minus[j], l_plus[j] + 1)]
            for j in range(2)
        ]
        number_bandwidths = len(bandwidths[0]) * len(bandwidths[1])
    elif agg_type == "ksd":
        max_samples = 500
        distances = scipy.spatial.distance.pdist(X[:max_samples], "euclidean")  
        distances = distances[distances > 0]
        lambda_min = 1
        lambda_max = np.maximum(np.max(distances), 2)
        power = (lambda_max / lambda_min) ** (1 / (number_bandwidths - 1))
        bandwidths = np.array([power ** i * lambda_min / X.shape[1] for i in range(number_bandwidths)])

    # compute all h-values
    h_values, index_i, index_j, N = compute_h_values(
        X, Y, R, bandwidths, return_indices_N=True
    )
    
    # compute bootstrap and original statistics
    bootstrap_values, original_value = compute_bootstrap_values(
        h_values, index_i, index_j, N, B1 + B2, seed, batch_size, return_original=True, memory_percentage=memory_percentage
    )
    
    # compute quantile
    quantiles, u_correction = compute_quantile(
        bootstrap_values, original_value, B1, B2, B3, alpha, weights_type
    )
    
    # compute test output and dictionary
    output = return_test_output(
        agg_type, 
        bootstrap_values, 
        original_value, 
        quantiles, 
        u_correction, 
        bandwidths, 
        weights_type, 
        B1, 
        return_dictionary,
    )
    
    return output


def inc(
    agg_type, 
    X, 
    Y, 
    R=200,
    alpha=0.05,
    batch_size="auto",
    memory_percentage=0.8,
    B=500, 
    seed=42,
    return_dictionary=False,
    bandwidth=None,
):
    """
    Efficient test for two-sample (MMD), independence (HSIC) and goodness-of-fit 
    (KSD) testing, using median bandwidth (no aggregation).
    
    Given the appropriate data for the type of testing, 
    return 0 if the test fails to reject the null (i.e. same distribution, independent, fits the data), 
    or return 1 if the test rejects the null (i.e. different distribution, dependent, does not fit the data).
    
    Parameters
    ----------
    agginc: str
        "mmd" or "hsic" or "ksd"
    X : array_like
        The shape of X must be of the form (N_X, d_X) where N_X is the number
        of samples and d_X is the dimension.
    Y : array_like
        The shape of Y must be of the form (N_Y, d_Y) 
        where N_Y is the number of samples and d_Y is the dimension.
        Case agginc = "mmd": Y is the second sample, we must have d_X = d_Y.
        Case agginc = "hsic": Y is the paired sample, we must have N_X = N_Y.
        Case agginc = "ksd": Y is the score of X, we must have N_Y = N_X and d_Y = d_X.
    R : int
        Number of superdiagonals to consider. 
        If R >= min(N_X, N_Y) - 1 then the complete U-statistic is computed in quadratic time.
    alpha: float
        The level alpha must be between 0 and 1.
    batch_size : int or None or str
        The memory cost consists in storing an array of shape (batch_size, R * N - R * (R - 1) / 2)
        where batch_size is between 1 and B.
        Using batch_size = "auto", calculates automatically the batch_size which uses 80% of the memory.
        For faster runtimes but using more memory, use batch_size = None (equivalent to batch_size = B)
        By decreasing batch_size from B, the memory cost is reduced but the runtimes increase.
    memory_percentage: float
        The value of memory_percentage must be between 0 and 1.
        It is used when batch_size = "auto", the batch_size is calculated automatically 
        to use memory_percentage of the memory.
    B: int
        B is the number of wild bootstrap samples to approximate the quantiles.
    seed: int 
        Random seed used for the randomness of the Rademacher variables.
    return_dictionary: bool
        If true, a dictionary is also returned containing the test out, the kernel, the bandwidth, 
        the statistic, the statistic quantile, the p-value and the p-value threshold value (level).
   bandwidth: float or list or None
        If bandwidths is None, the bandwidth used is the median heuristic.
        Otherwise, the bandwidth provided is used instead.
        If agg_type is "mmd" or "ksd", then bandwidth needs to be a float.
        If agg_type is "hsic", then bandwidths should be a list 
        containing 2 floats (bandwidths for X and Y).

        
    Returns
    -------
    output : int
        0 if the Inc test fails to reject the null (i.e. same distribution, independent, fits the data), 
        1 if the Inc test rejects the null (i.e. different distribution, dependent, does not fit the data).
    dictionary: dict
        Returned only if return_dictionary is True.
        Dictionary containing the output of the Inc test, the kernel, the bandwidth, 
        the statistic, the quantile, the p-value and the p-value threshold (level).
    
    Examples
    --------
    
    >>> # MMDInc
    >>> rs = np.random.RandomState(0)
    >>> X = rs.randn(500, 10)
    >>> Y = rs.randn(500, 10) + 1
    >>> output = inc("mmd", X, Y)
    >>> output
    1
    >>> output, dictionary = inc("mmd", X, Y, return_dictionary=True)
    >>> output
    1
    >>> dictionary
    {'MMDInc test reject': True,
     'Kernel': 'Gaussian',
     'Bandwidth': 5.286007689131921,
     'MMD': 0.28101008856766124,
     'MMD quantile': 0.002611614394879011,
     'p-value': 0.001996007984031936,
     'p-value threshold': 0.05}
    
    >>> # HSICInc
    >>> rs = np.random.RandomState(0)
    >>> X = rs.randn(500, 10)
    >>> Y = 0.5 * X + rs.randn(500, 10)
    >>> output = inc("hsic", X, Y)
    >>> output
    1
    >>> output, dictionary = inc("hsic", X, Y, return_dictionary=True)
    >>> output
    1
    >>> dictionary
    {'HSICInc test reject': True,
     'Kernel': 'Gaussian',
     'Bandwidth X': 4.276330859630623,
     'Bandwidth Y': 4.783357279137176,
     'HSIC': 0.007071571325159379,
     'HSIC quantile': 0.0004403802401235617,
     'p-value': 0.001996007984031936,
     'p-value threshold': 0.05}
    
   >>> # KSDInc
    >>> perturbation = 0.5
    >>> rs = np.random.RandomState(0)
    >>> X = rs.gamma(5 + perturbation, 5, (500, 1))
    >>> score_gamma = lambda x, k, theta : (k - 1) / x - 1 / theta
    >>> score_X = score_gamma(X, 5, 5)
    >>> output = inc("ksd", X, score_X)
    >>> output
    1
    >>> output, dictionary = inc("ksd", X, score_X, return_dictionary=True)
    >>> output
    1
    >>> dictionary
    {'KSDInc test reject': True,
     'Kernel': 'imq',
     'Bandwidth': 10.187002895596237,
     'KSD': 2.4671437226767908e-05,
     'KSD quantile': 5.920405142843872e-06,
     'p-value': 0.001996007984031936,
     'p-value threshold': 0.05}
    """
    
    # function compute_h_values
    if agg_type == "mmd":
        compute_h_values = compute_h_MMD_values
    elif agg_type == "hsic":
        compute_h_values = compute_h_HSIC_values
    elif agg_type == "ksd":
        compute_h_values = compute_h_KSD_values
    else:
        raise ValueError('The value of agg_type should be "mmd" or' '"hsic" or "ksd".')
        
    # bandwidth: use provided one or compute median heuristic
    if bandwidth is not None:
        if agg_type == "mmd" or agg_type == "ksd":
            bandwidths = np.array(bandwidth).reshape(1)
        elif agg_type == "hsic":
            assert len(bandwidth) == 2
            bandwidths = (np.array(bandwidth[0]).reshape(1), np.array(bandwidth[1]).reshape(1))
    elif agg_type == "mmd":
        max_samples = 500
        distances = scipy.spatial.distance.cdist(X[:max_samples], Y[:max_samples], "euclidean").reshape(-1)
        median_bandwidth = np.median(distances[distances > 0])
        bandwidths = np.array([median_bandwidth])
    elif agg_type == "hsic":
        max_samples = 500
        distances = scipy.spatial.distance.pdist(X[:max_samples], "euclidean") 
        median_bandwidth_X = np.median(distances[distances > 0])
        distances = scipy.spatial.distance.pdist(Y[:max_samples], "euclidean")  
        median_bandwidth_Y = np.median(distances[distances > 0])
        bandwidths = (np.array([median_bandwidth_X]), np.array([median_bandwidth_Y]))
    elif agg_type == "ksd":
        max_samples = 500
        distances = scipy.spatial.distance.pdist(X[:max_samples], "euclidean")  
        median_bandwidth = np.median(distances[distances > 0])
        bandwidths = np.array([median_bandwidth])
    else:
        raise ValueError('The value of agg_type should be "mmd" or "hsic" or "ksd".')
        
    # compute all h-values
    h_values, index_i, index_j, N = compute_h_values(
        X, Y, R, bandwidths, return_indices_N=True
    )
    
    # compute bootstrap and original statistics
    bootstrap_values, original_value = compute_bootstrap_values(
        h_values, index_i, index_j, N, B, seed, batch_size, return_original=True, memory_percentage=memory_percentage
    )
    original_value = original_value[0]
    
    # compute quantile
    assert bootstrap_values.shape[0] == 1
    bootstrap_1 = np.column_stack([bootstrap_values, original_value])
    bootstrap_1_sorted = np.sort(bootstrap_1)
    quantile = bootstrap_1_sorted[0, int(np.ceil((B + 1) * (1 - alpha))) - 1]
    
    # reject if p_val <= alpha
    p_val = np.mean((bootstrap_1 - original_value.reshape(-1, 1) >= 0), -1)[0]
    reject_p_val = p_val <= alpha

    # reject if original_value > quantile
    reject_stat_val = original_value > quantile

    # assert both rejection methods are equivalent
    assert reject_p_val == reject_stat_val

    # create rejection dictionary 
    reject_dictionary = {}
    reject_dictionary[agg_type.upper() + "Inc test reject"] = reject_p_val
    reject_dictionary["Kernel"] = "imq" if agg_type == "ksd" else "Gaussian"
    if agg_type == "hsic":
        reject_dictionary["Bandwidth X"] = bandwidths[0][0]
        reject_dictionary["Bandwidth Y"] = bandwidths[1][0]
    else:
        reject_dictionary["Bandwidth"] = bandwidths[0]
    reject_dictionary[agg_type.upper()] = original_value
    reject_dictionary[agg_type.upper() + " quantile"] = quantile
    reject_dictionary["p-value"] = p_val
    reject_dictionary["p-value threshold"] = alpha

    # return output and dictionary
    if return_dictionary:
        return int(reject_dictionary[agg_type.upper() + "Inc test reject"]), reject_dictionary
    else:
        return int(reject_dictionary[agg_type.upper() + "Inc test reject"])


def create_indices(N, R):
    """
    Return lists of indices of R subdiagonals of N x N matrix
    """
    index_X = list(
        itertools.chain(*[[i for i in range(N - r)] for r in range(1, R + 1)])
    )
    index_Y = list(
        itertools.chain(*[[i + r for i in range(N - r)] for r in range(1, R + 1)])
    )
    return index_X, index_Y


def compute_h_MMD_values(X, Y, R, bandwidths, return_indices_N=False):
    """
    Compute h_MMD values.

    inputs:
        X (m,d)
        Y (n,d)
        R int
        bandwidths (#bandwidths,)

    output (#bandwidths, R * N - R * (R - 1) / 2)
    """
    N = min(X.shape[0], Y.shape[0])
    assert X.shape[1] == Y.shape[1]

    index_i, index_j = create_indices(N, R)

    norm_Xi_Xj = np.linalg.norm(X[index_i] - X[index_j], axis=1) ** 2
    norm_Xi_Yj = np.linalg.norm(X[index_i] - Y[index_j], axis=1) ** 2
    norm_Yi_Xj = np.linalg.norm(Y[index_i] - X[index_j], axis=1) ** 2
    norm_Yi_Yj = np.linalg.norm(Y[index_i] - Y[index_j], axis=1) ** 2
    
    h_values = np.zeros((bandwidths.shape[0], norm_Xi_Xj.shape[0]))
    for r in range(bandwidths.shape[0]):
        K_Xi_Xj_b = np.exp(-norm_Xi_Xj / bandwidths[r] ** 2)
        K_Xi_Yj_b = np.exp(-norm_Xi_Yj / bandwidths[r] ** 2)
        K_Yi_Xj_b = np.exp(-norm_Yi_Xj / bandwidths[r] ** 2)
        K_Yi_Yj_b = np.exp(-norm_Yi_Yj / bandwidths[r] ** 2)
        h_values[r] = K_Xi_Xj_b - K_Xi_Yj_b - K_Yi_Xj_b + K_Yi_Yj_b

    if return_indices_N:
        return h_values, index_i, index_j, N
    else:
        return h_values


def compute_h_HSIC_values(X, Y, R, bandwidths, return_indices_N=False):
    """
    Compute h_HSIC values.
    
    inputs:
        X (m,d)
        Y (m,d)
        R int
        bandwidths (2, #bandwidths) (bandwidths for X and for Y)
        
    output (#bandwidths, R * N - R * (R - 1) / 2)
    """
    assert X.shape[0] == Y.shape[0]
    N = int(X.shape[0] / 2)

    bandwidths_X = np.array(bandwidths[0])
    bandwidths_Y = np.array(bandwidths[1])

    h_X_values, index_i, index_j, Nbis = compute_h_MMD_values(
        X[:N], X[N:], R, bandwidths_X, True
    )
    assert N == Nbis
    h_Y_values = compute_h_MMD_values(Y[:N], Y[N:], R, bandwidths_Y)

    # we need to consider all pairs of bandwidths
    h_XY_values = (
        np.expand_dims(h_X_values, 0) * np.expand_dims(h_Y_values, 1)
    ).reshape(h_X_values.shape[0] * h_Y_values.shape[0], -1)

    if return_indices_N:
        return h_XY_values, index_i, index_j, N
    else:
        return h_XY_values


def compute_h_KSD_values(
    X, score_X, R, bandwidths, return_indices_N=False, beta_imq=-0.5
):
    """
    Compute h_KSD values.

    inputs:
        X (m,d)
        Y (m,d)
        R int
        bandwidths (#bandwidths,)
        beta_imq in (-1, 0)

    output (#bandwidths, R * N - R * (R - 1) / 2)
    """
    assert X.shape == score_X.shape
    N, d = X.shape

    index_i, index_j = create_indices(N, R)

    Xi_minus_Xj = X[index_i] - X[index_j]
    norm_Xi_Xj = np.linalg.norm(Xi_minus_Xj, axis=1) ** 2
    sXi = score_X[index_i]
    sXj = score_X[index_j]
    sXi_minus_sXj = sXi - sXj
    sXi_minus_sXj_dot_Xi_minus_Xj = np.einsum(
        "ij,ij->i", sXi_minus_sXj, Xi_minus_Xj, optimize=True
    )
    sXi_dot_sXj = np.einsum("ij,ij->i", sXi, sXj, optimize=True)

    h_values = np.zeros((bandwidths.shape[0], Xi_minus_Xj.shape[0]))
    for r in range(bandwidths.shape[0]):
        b_norm_Xi_Xj = bandwidths[r] ** 2 + norm_Xi_Xj
        h_values[r] = (
            sXi_dot_sXj * b_norm_Xi_Xj**beta_imq
            - sXi_minus_sXj_dot_Xi_minus_Xj
            * 2
            * beta_imq
            * b_norm_Xi_Xj ** (beta_imq - 1)
            - 2 * d * beta_imq * b_norm_Xi_Xj ** (beta_imq - 1)
            - 4
            * beta_imq
            * (beta_imq - 1)
            * norm_Xi_Xj
            * b_norm_Xi_Xj ** (beta_imq - 2)
        )

    if return_indices_N:
        return h_values, index_i, index_j, N
    else:
        return h_values


def compute_bootstrap_values(
    h_values, index_i, index_j, N, B, seed, batch_size="auto", return_original=False, memory_percentage=0.8
):
    """
    Compute B bootstrap values.

    inputs:
        h_values, index_i, index_j = compute_h_XXX_values(...)
        h_values (#bandwidths, R * N - R * (R - 1) / 2)
        N int
        B int
        seed int


    output (#bandwidths, B)
    """
    # Numpy
    rs = np.random.RandomState(seed)
    epsilon = rs.choice([1.0, -1.0], size=(N, B))
    
    # Bootstrap values can be computed as follows
    # with memory cost of storing an array (e_values) 
    # of size (R * N - R * (R - 1) / 2, B)
    
    # e_values = epsilon[index_i] * epsilon[index_j]
    # bootstrap_values = h_values @ e_values
    
    # Instead we use batches to store only arrays
    # of size (R * N - R * (R - 1) / 2, batch_size)
    # where batch_size is automatically chosen to use 80% of the memory
    # In the experiments of the paper, batch_size = None (i.e. batch_size = B) has been used
    # Larger batch_size increases the memory cost and decreases computational time
    
    if batch_size == None:
        batch_size = B
    elif batch_size == "auto":
        # Automatically compute the batch size depending on cpu memory 
        memory_cpu = psutil.virtual_memory().total # bytes
        memory_single_array = np.zeros(h_values.shape[0]).nbytes
        batch_size = int(memory_cpu * memory_percentage / memory_single_array)
    bootstrap_values = np.zeros((h_values.shape[0], epsilon.shape[1]))
    i = 0
    index = 0
    while index + batch_size < B or i == 0:
        index = i * batch_size
        epsilon_b = epsilon[:, index : index + batch_size]
        e_values_b = epsilon_b[index_i] * epsilon_b[index_j]
        bootstrap_values[:, index : index + batch_size] = h_values @ e_values_b
        i += 1
    bootstrap_values = bootstrap_values / len(index_i)

    if return_original:
        original_value = h_values @ np.ones(h_values.shape[1]) / len(index_i)
        return bootstrap_values, original_value
    else:
        return bootstrap_values


def compute_quantile(bootstrap_values, original_value, B1, B2, B3, alpha, weights_type):
    """
    Compute quantile.

    inputs:
        bootstrap_values (#bandwidths, B1 + B2)
        original_value (#bandwidths, )
        B1 int
        B2 int
        B3 int
        alpha in (0,1)
        weights_type "uniform" or "decreasing" or "increasing" or "centred"

    returns quantiles (#bandwidths, )
    """
    # this can be adjusted B1 for quantiles, B2 for P_u
    bootstrap_1 = np.column_stack([bootstrap_values[:, :B1], original_value])
    bootstrap_1_sorted = np.sort(bootstrap_1)  # sort each row
    bootstrap_2 = bootstrap_values[:, B1:]
    assert B2 == bootstrap_2.shape[1]

    weights = create_weights(bootstrap_values.shape[0], weights_type)
    # (1-u*w_lambda)-quantiles for the #bandwidths
    quantiles = np.zeros((bootstrap_values.shape[0], 1))
    u_min = 0.0
    u_max = np.min(1 / weights)
    for _ in range(B3):
        u = (u_max + u_min) / 2
        for i in range(bootstrap_values.shape[0]):
            quantiles[i] = bootstrap_1_sorted[
                i, int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1
            ]
        P_u = np.sum(np.max(bootstrap_2 - quantiles, 0) > 0) / B2
        if P_u <= alpha:
            u_min = u
        else:
            u_max = u
    u = u_min
    for i in range(bootstrap_values.shape[0]):
        quantiles[i] = bootstrap_1_sorted[
            i, int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1
        ]
    return quantiles, u

def return_test_output(agg_type, bootstrap_values, original_value, quantiles, u_correction, bandwidths, weights_type, B1, return_dictionary):
    """
    Compute test output and dictionary.
    
    inputs:
        agg_type "mmd" or "hsic" or "ksd"
        bootstrap_values (#bandwidths, B1 + B2)
        original_value (#bandwidths, )
        quantiles (#bandwidths, )
        u_correction float
        bandwidths (#bandwidths, ) for agg_type "mmd" or "hsic", 
                   list for agg_type "hsic" with first and second elements bandwidths_X and bandwidths_Y
        weights_type "uniform" or "decreasing" or "increasing" or "centred"
        B1 int
        return_dictionary bool
        
    returns 0 or 1 depending on test output
    also returns dictionary if return_dictionary is True
    """
    
    bootstrap_1 = np.column_stack([bootstrap_values[:, :B1], original_value])
    
    number_bandwidths = bootstrap_values.shape[0]
    weights = create_weights(number_bandwidths, weights_type)
    
    p_vals = np.mean((bootstrap_1 - original_value.reshape(-1, 1) >= 0), -1)
    thresholds = u_correction * weights
    # reject if p_val <= threshold
    reject_p_vals = p_vals <= thresholds

    stat_vals = original_value
    quantiles = quantiles.reshape(-1)
    # reject if stat_val > quantile
    reject_stat_vals = stat_vals > quantiles

    # assert both rejection methods are equivalent
    np.testing.assert_array_equal(reject_p_vals, reject_stat_vals)
    
    if agg_type == "hsic":
        bandwidths = [(bandwidths[0][i], bandwidths[1][j]) for j in range(len(bandwidths[1])) for i in range(len(bandwidths[0]))]
        assert number_bandwidths == len(bandwidths)

    # create rejection dictionary 
    reject_dictionary = {}
    reject_dictionary[agg_type.upper() + "AggInc test reject"] = False
    for i in range(number_bandwidths):
        index = "Single test " + str(i + 1)
        reject_dictionary[index] = {}
        reject_dictionary[index]["Reject"] = reject_p_vals[i]
        reject_dictionary[index]["Kernel"] = "imq" if agg_type == "ksd" else "Gaussian"
        if agg_type == "hsic":
            reject_dictionary[index]["Bandwidth X"] = bandwidths[i][0]
            reject_dictionary[index]["Bandwidth Y"] = bandwidths[i][1]
        else:
            reject_dictionary[index]["Bandwidth"] = bandwidths[i]
        reject_dictionary[index][agg_type.upper()] = stat_vals[i]
        reject_dictionary[index][agg_type.upper() + " quantile"] = quantiles[i]
        reject_dictionary[index]["p-value"] = p_vals[i]
        reject_dictionary[index]["p-value threshold"] = thresholds[i]
        # Aggregated test rejects if one single test rejects
        reject_dictionary[agg_type.upper() + "AggInc test reject"] = any((
            reject_dictionary[agg_type.upper() + "AggInc test reject"], 
            reject_p_vals[i]
        ))

    if return_dictionary:
        return int(reject_dictionary[agg_type.upper() + "AggInc test reject"]), reject_dictionary
    else:
        return int(reject_dictionary[agg_type.upper() + "AggInc test reject"])


def create_weights(N, weights_type):
    """
    Create weights as defined in Section 5.1 of MMD Aggregated Two-Sample Test (Schrab et al., 2021).
    inputs: N: number of bandwidths to test
            weights_type: "uniform" or "decreasing" or "increasing" or "centred"
    output: (N,) array of weights
    """
    if weights_type == "uniform":
        weights = np.array(
            [
                1 / N,
            ]
            * N
        )
    elif weights_type == "decreasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / (i * normaliser) for i in range(1, N + 1)])
    elif weights_type == "increasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / ((N + 1 - i) * normaliser) for i in range(1, N + 1)])
    elif weights_type == "centred":
        if N % 2 == 1:
            normaliser = sum([1 / (abs((N + 1) / 2 - i) + 1) for i in range(1, N + 1)])
            weights = np.array(
                [1 / ((abs((N + 1) / 2 - i) + 1) * normaliser) for i in range(1, N + 1)]
            )
        else:
            normaliser = sum(
                [1 / (abs((N + 1) / 2 - i) + 0.5) for i in range(1, N + 1)]
            )
            weights = np.array(
                [
                    1 / ((abs((N + 1) / 2 - i) + 0.5) * normaliser)
                    for i in range(1, N + 1)
                ]
            )
    else:
        raise ValueError(
            'The value of weights_type should be "uniform" or'
            '"decreasing" or "increasing" or "centred".'
        )
    return weights
