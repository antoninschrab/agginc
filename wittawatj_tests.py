"""
The tests in this file are taken from the implementation of Wittawat Jitkrittum (https://github.com/wittawatj/).

- Tests: ME (Mean Embedding) and SCF (Smooth Characteristic Function)
- Paper: [Interpretable Distribution Features with Maximum Testing Power](https://proceedings.neurips.cc/paper/2016/file/0a09c8844ba8f0936c20bd791130d6b6-Paper.pdf)
- Authors: Wittawat Jitkrittum, Zoltán Szabó, Kacper Chwialkowski, Arthur Gretton
- Code: [interpretable-test repository](https://github.com/wittawatj/interpretable-test) by [Wittawat Jitkrittum](https://github.com/wittawatj)

- Test: FSIC (Finite Set Independence Criterion)
- Paper: [An Adaptive Test of Independence with Analytic Kernel Embeddings](http://proceedings.mlr.press/v70/jitkrittum17a/jitkrittum17a.pdf)
- Authors: Wittawat Jitkrittum, Zoltán Szabó, Arthur Gretton
- Code: [fsic-test repository](https://github.com/wittawatj/fsic-test) by [Wittawat Jitkrittum](https://github.com/wittawatj)

- Test: FSSD (Finite Set Stein Discrepancy)
- Paper: [A Linear-Time Kernel Goodness-of-Fit Test](https://papers.nips.cc/paper/2017/file/979d472a84804b9f647bc185a877a8b5-Paper.pdf)
- Authors: Wittawat Jitkrittum, Wenkai Xu, Zoltán Szabó, Kenji Fukumizu, Arthur Gretton
- Code: [kernel-gof repository](https://github.com/wittawatj/kernel-gof) by [Wittawat Jitkrittum](https://github.com/wittawatj)
"""


from fsic.indtest import GaussNFSIC as fsic_GaussNFSIC
from fsic.data import PairedData as fsic_PairedData
import logging


# based on function job_nfsicJ10_stoopt()
# https://github.com/wittawatj/fsic-test/blob/master/fsic/ex/ex2_prob_params.py
def nfsic(X, Y, r, J=10, n_permute=500, alpha=0.05):
    pdata = fsic_PairedData(X, Y)
    tr, te = pdata.subsample(X.shape[0], seed=r+4).split_tr_te(tr_proportion=0.5, seed=r+5)
    nfsic_opt_options = {'n_test_locs':J, 'max_iter':200,
    'V_step':1, 'W_step':1, 'gwidthx_step':1, 'gwidthy_step':1,
    'batch_proportion':0.7, 'tol_fun':1e-4, 'step_pow':0.5, 'seed':r+2,
    'reg': 1e-6}
    op_V, op_W, op_gwx, op_gwy, info = fsic_GaussNFSIC.optimize_locs_widths(tr,
            alpha, **nfsic_opt_options )
    nfsic_opt = fsic_GaussNFSIC(op_gwx, op_gwy, op_V, op_W, alpha, reg='auto', n_permute=n_permute, seed=r+3)
    nfsic_opt_result  = nfsic_opt.perform_test(te)
    return int(nfsic_opt_result['h0_rejected'])


from kgof.goftest import FSSDH0SimCovObs as kgof_FSSDH0SimCovObs
from kgof.util import fit_gaussian_draw as kgof_fit_gaussian_draw
from kgof.goftest import IMQFSSD as kgof_IMQFSSD
from kgof.kernel import KIMQ as kgof_KIMQ
from kgof.goftest import FSSD as kgof_FSSD
from kgof.data import Data as kgof_Data


# based on function job_fssdJ1q_imq_optv()
# https://github.com/wittawatj/kernel-gof/blob/master/kgof/ex/ex1_vary_n.py
def fssd(X, p, r, J=10, alpha=0.05):
    """
    FSSD with optimization on tr. Test on te. Use an inverse multiquadric
    kernel (IMQ). Optimize only the test locations (V). Fix the kernel
    parameters to b = -0.5, c=1. These are the recommended values from
        Measuring Sample Quality with Kernels
        Jackson Gorham, Lester Mackey
    """
    
    null_sim = kgof_FSSDH0SimCovObs(n_simulate=2000, seed=r)
    
    data = kgof_Data(X)
    tr, te = data.subsample(X.shape[0], seed=r+4).split_tr_te(tr_proportion=0.5, seed=r+5)
        
    Xtr = tr.data()
    # IMQ kernel parameters
    b = -0.5
    c = 1.0

    # fit a Gaussian to the data and draw to initialize V0
    V0 = kgof_fit_gaussian_draw(Xtr, J, seed=r+1, reg=1e-6)

    ops = {
        'reg': 1e-2,
        'max_iter': 40,
        'tol_fun': 1e-4,
        'disp': False,
        'locs_bounds_frac':10.0,
        }

    V_opt, info = kgof_IMQFSSD.optimize_locs(p, tr, b, c, V0, **ops) 
    k_imq = kgof_KIMQ(b=b, c=c)

    # Use the optimized parameters to construct a test
    fssd_imq = kgof_FSSD(p, k_imq, V_opt, null_sim=null_sim, alpha=alpha)
    fssd_imq_result = fssd_imq.perform_test(te)

    return int(fssd_imq_result['h0_rejected'])


from freqopttest.tst import MeanEmbeddingTest as fot_MeanEmbeddingTest
from freqopttest.data import TSTData as fot_TSTData


# based on job_met_opt() function
# https://github.com/wittawatj/interpretable-test/blob/master/freqopttest/ex/ex1_power_vs_n.py
def met(X, Y, r, J=10, alpha=0.05):
    """MeanEmbeddingTest with test locations optimzied.
    Return results from calling perform_test()"""
    # MeanEmbeddingTest. optimize the test locations
    
    assert X.shape[0] == Y.shape[0]
    data = fot_TSTData(X, Y)
    tr, te = data.subsample(X.shape[0], seed=r+4).split_tr_te(tr_proportion=0.5, seed=r+5)

    met_opt_options = {'n_test_locs': J, 'max_iter': 200, 
            'locs_step_size': 0.1, 'gwidth_step_size': 0.1, 'seed': r+92856,
            'tol_fun': 1e-3}
    test_locs, gwidth, info = fot_MeanEmbeddingTest.optimize_locs_width(tr, alpha, **met_opt_options)
    met_opt = fot_MeanEmbeddingTest(test_locs, gwidth, alpha)
    met_opt_test  = met_opt.perform_test(te)

    return int(met_opt_test['h0_rejected'])


from freqopttest.tst import SmoothCFTest as fot_SmoothCFTest


# based on job_scf_opt() function 
# https://github.com/wittawatj/interpretable-test/blob/master/freqopttest/ex/ex1_power_vs_n.py
def scf(X, Y, r, J=10, alpha=0.05):
    """SmoothCFTest with frequencies optimized."""
    
    assert X.shape[0] == Y.shape[0]
    data = fot_TSTData(X, Y)
    tr, te = data.subsample(X.shape[0], seed=r+4).split_tr_te(tr_proportion=0.5, seed=r+5)
    
    op = {'n_test_freqs': J, 'max_iter': 200, 'freqs_step_size': 0.1, 
            'gwidth_step_size': 0.1, 'seed': r+92856, 'tol_fun': 1e-3}
    test_freqs, gwidth, info = fot_SmoothCFTest.optimize_freqs_width(tr, alpha, **op)
    scf_opt = fot_SmoothCFTest(test_freqs, gwidth, alpha)
    scf_opt_test = scf_opt.perform_test(te)

    return int(scf_opt_test['h0_rejected'])