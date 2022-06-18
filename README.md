# Code for AggInc: Computationally Efficient Aggregated Kernel Tests using Incomplete U-statistics

This anonymised GitHub repository contains the code for our paper Computationally Efficient Aggregated Kernel Tests using Incomplete U-statistics.
The three tests we have proposed 

- MMDAggInc
- HSICAggInc
- KSDAggInc

are implemented as the function `agginc` in [agginc_test.py](agginc_test.py).
The code for reproducibility of the experiments of our paper, and for generating the figures in [figures](figures), is presented in the notebook [experiments.ipynb](experiments.ipynb). The outputs of all the experiments are saved in [results](results).

## Requirements
- `python 3.9`

For our AggInc tests, we only require the `numpy` and `scipy` packages. All other packages in [requirements.txt](requirements.txt) are required to run the tests we are comparing against. The `numpy` package version `<= 1.21` is only needed for compatibility with the `theano` package.

## Installation

In a chosen directory, clone this repository and change to its directory.
 
We then recommend creating and activating a virtual environment by either 
- using `venv`:
  ```
  python3 -m venv agginc-env
  source agginc-env/bin/activate
  # can be deactivated by running:
  # deactivate
  ```
- or using `conda`:
  ```
  conda create --name agginc-env python=3.9
  conda activate agginc-env
  # can be deactivated by running:
  # conda deactivate
  ```
The required packages can then be installed in the virtual environment by running
```
python -m pip install -r requirements.txt
```

## References

In our experiments, we compare MMDAggInc to
- Tests: ME (Mean Embedding) and SCF (Smooth Characteristic Function)
- Paper: [Interpretable Distribution Features with Maximum Testing Power](https://proceedings.neurips.cc/paper/2016/file/0a09c8844ba8f0936c20bd791130d6b6-Paper.pdf)
- Authors: Wittawat Jitkrittum, Zoltán Szabó, Kacper Chwialkowski, Arthur Gretton
- Code: [interpretable-test
 repository](https://github.com/wittawatj/interpretable-test) by [Wittawat Jitkrittum](https://github.com/wittawatj)

and to

- Test: PSI OST (Post Selection Inference One-Sided Test)
- Paper: [Learning Kernel Tests Without Data Splitting](https://proceedings.neurips.cc/paper/2020/file/44f683a84163b3523afe57c2e008bc8c-Paper.pdf)
- Authors: Jonas M. Kübler, Wittawat Jitkrittum, Bernhard Schölkopf, Krikamol Muandet
- Code: [tests-wo-splitting repository](https://github.com/MPI-IS/tests-wo-splitting) by [Max Planck Institute for Intelligent Systems](https://github.com/MPI-IS)

We compare HSICAggInc to 
- Test: FSIC (Finite Set Independence Criterion)
- Paper: [An Adaptive Test of Independence with Analytic Kernel Embeddings](http://proceedings.mlr.press/v70/jitkrittum17a/jitkrittum17a.pdf)
- Authors: Wittawat Jitkrittum, Zoltán Szabó, Arthur Gretton
- Code: [fsic-test
 repository](https://github.com/wittawatj/fsic-test) by [Wittawat Jitkrittum](https://github.com/wittawatj)

We compare KSDAggInc to 
- Test: FSSD (Finite Set Stein Discrepancy)
- Paper: [A Linear-Time Kernel Goodness-of-Fit Test](https://papers.nips.cc/paper/2017/file/979d472a84804b9f647bc185a877a8b5-Paper.pdf)
- Authors: Wittawat Jitkrittum, Wenkai Xu, Zoltán Szabó, Kenji Fukumizu, Arthur Gretton
- Code: [kernel-gof repository](https://github.com/wittawatj/kernel-gof) by [Wittawat Jitkrittum](https://github.com/wittawatj)

and to

- Test: LSD (Learning Stein Discrepancy)
- Paper: [Learning the Stein Discrepancy
for Training and Evaluating Energy-Based Models without Sampling](http://proceedings.mlr.press/v119/grathwohl20a/grathwohl20a.pdf)
- Authors: Will Grathwohl, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, Richard Zemel
- Code: [LSD repository](https://github.com/wgrathwohl/LSD) by [Will Grathwohl](https://github.com/wgrathwohl)

## License

MIT License (see [LICENSE.md](LICENSE.md)).
