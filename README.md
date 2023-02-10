# AggInc package

This package implements MMDAggInc, HSICAggInc and KSDAggInc tests for two-sample, independence and goodness-of-fit testing, as proposed in our paper [Efficient Aggregated Kernel Tests using Incomplete U-statistics](https://arxiv.org/pdf/2206.09194.pdf).
The experiments of the paper can be reproduced using the [agginc-paper](https://github.com/antoninschrab/agginc-paper/) repository.
The package contains implementations both in Numpy and in Jax, we recommend using the Jax version as it runs more than 100 times faster after compilation (results from the notebook [speed.ipynb](https://github.com/antoninschrab/agginc-paper/blob/master/speed.ipynb) in the [agginc-paper](https://github.com/antoninschrab/agginc-paper/) repository).

| Speed in ms | Numpy (CPU) | Jax (CPU) | Jax (GPU) |
| -- | -- | -- | -- |
| MMDAggInc | 4490 | 844 | 23 |
| HSICAggInc | 2820 | 539 | 18 |
| KSDAggInc | 3770 | 590 | 22 |

We provide installation instructions and example code below showing how to use our MMDAggInc, HSICAggInc and KSDAggInc tests.
We also provide a demo notebook [demo.ipynb](https://github.com/antoninschrab/agginc-paper/blob/master/demo.ipynb) in the [agginc-paper](https://github.com/antoninschrab/agginc-paper/) repository.

## Requirements

The requirements for the Numpy version are:
- `python 3.9`
  - `numpy`
  - `scipy`
  - `psutil`
  - `gputil`  

The requirements for the Jax version are:
- `python 3.9`
  - `jax`
  - `jaxlib`
  - `psutil`
  - `gputil`  

## Installation

First, we recommend creating a conda environment:
```bash
conda create --name agginc-env python=3.9
conda activate agginc-env
# can be deactivated by running:
# conda deactivate
```

We then install the required depedencies by running either:
- for GPU:
  ```bash
  conda install -c conda-forge -c nvidia pip numpy scipy cuda-nvcc "jaxlib=0.4.1=*cuda*" jax psutil gputil
  ```
- or, for CPU:
  ```bash
  conda install -c conda-forge -c nvidia pip numpy scipy cuda-nvcc jaxlib=0.4.1 jax psutil gputil
  ```
  
Our `agginc` package can then be installed as follows:
```bash
pip install git+https://github.com/antoninschrab/agginc.git
```

## MMDAggInc

**Two-sample testing:** Given arrays X of shape $(N_X, d)$ and Y of shape $(N_Y, d)$, our MMDAggInc test `agginc("mmd", X, Y)` returns 0 if the samples X and Y are believed to come from the same distribution, and 1 otherwise.

**Jax compilation:** The first time the function is evaluated, Jax compiles it. 
After compilation, it can fastly be evaluated at any other X and Y of the same shape. 
If the function is given arrays with new shapes, the function is compiled again.
For details, check out the [demo.ipynb](https://github.com/antoninschrab/agginc-paper/blob/master/demo.ipynb) and [speed.ipynb](https://github.com/antoninschrab/agginc-paper/blob/master/speed.ipynb) notebooks in the [agginc-paper](https://github.com/antoninschrab/agginc-paper/) repository.

```python
# import modules
>>> import numpy as np 
>>> import jax.numpy as jnp
>>> from agginc import agginc, human_readable_dict # jax version
>>> # from agginc.np import agginc

# generate data for two-sample test
>>> key = random.PRNGKey(0)
>>> key, subkey = random.split(key)
>>> subkeys = random.split(subkey, num=2)
>>> X = random.uniform(subkeys[0], shape=(500, 10))
>>> Y = random.uniform(subkeys[1], shape=(500, 10)) + 1

# run MMDAggInc test
>>> output = agginc("mmd", X, Y)
>>> output
Array(1, dtype=int32)
>>> output.item()
1
>>> output, dictionary = agginc("mmd", X, Y, return_dictionary=True)
>>> output
Array(1, dtype=int32)
>>> human_readable_dict(dictionary)
>>> dictionary
{'MMDAggInc test reject': True,
 'Single test 1': {'Bandwidth': 0.8926196098327637,
  'Kernel Gaussian': True,
  'MMD': 0.3186362385749817,
  'MMD quantile': 0.0025616204366087914,
  'Reject': True,
  'p-value': 0.0019960079807788134,
  'p-value threshold': 0.04590817913413048},
  ...
}
```

## HSICAggInc

**Independence testing:** Given paired arrays X of shape $(N, d_X)$ and Y of shape $(N, d_Y)$, our HSICAggInc test `agginc("hsic", X, Y)` returns 0 if the paired samples X and Y are believed to be independent, and 1 otherwise.

**Jax compilation:** The first time the function is evaluated, Jax compiles it. 
After compilation, it can fastly be evaluated at any other X and Y of the same shape. 
If the function is given arrays with new shapes, the function is compiled again.
For details, check out the [demo.ipynb](https://github.com/antoninschrab/agginc-paper/blob/master/demo.ipynb) and [speed.ipynb](https://github.com/antoninschrab/agginc-paper/blob/master/speed.ipynb) notebooks in the [agginc-paper](https://github.com/antoninschrab/agginc-paper/) repository.

```python
# import modules
>>> import numpy as np 
>>> import jax.numpy as jnp
>>> from agginc import agginc, human_readable_dict # jax version
>>> # from agginc.np import agginc

# generate data for independence test 
>>> key = random.PRNGKey(0)
>>> key, subkey = random.split(key)
>>> subkeys = random.split(subkey, num=2)
>>> X = random.uniform(subkeys[0], shape=(500, 10))
>>> Y = 0.5 * X + random.uniform(subkeys[1], shape=(500, 10))

# run HSICAggInc test
>>> output = agginc("hsic", X, Y)
>>> output
Array(1, dtype=int32)
>>> output.item()
1
>>> output, dictionary = agginc("hsic", X, Y, return_dictionary=True)
>>> output
Array(1, dtype=int32)
>>> human_readable_dict(dictionary)
>>> dictionary
{'HSICAggInc test reject': True,
 'Single test 1': {'Bandwidth X': 0.31978243589401245,
  'Bandwidth Y': 0.3518877327442169,
  'HSIC': 3.8373030974980793e-07,
  'HSIC quantile': 8.487702416459797e-07,
  'Kernel Gaussian': True,
  'Reject': False,
  'p-value': 0.17365269362926483,
  'p-value threshold': 0.007984011434018612},
  ...
}
```

## KSDAggInc

**Goodness-of-fit testing:** Given arrays X and score_X both of shape $(N, d)$, where score_X is the score of X (i.e. $\nabla p(x)$ where $p$ is the model density), our KSDAggInc test `agginc("ksd", X, Y)` returns 0 if the samples X are believed to have been drawn from the density $p$, and 1 otherwise.

**Jax compilation:** The first time the function is evaluated, Jax compiles it. 
After compilation, it can fastly be evaluated at any other X and score_X of the same shape. 
If the function is given arrays with new shapes, the function is compiled again.
For details, check out the [demo.ipynb](https://github.com/antoninschrab/agginc-paper/blob/master/demo.ipynb) and [speed.ipynb](https://github.com/antoninschrab/agginc-paper/blob/master/speed.ipynb) notebooks in the [agginc-paper](https://github.com/antoninschrab/agginc-paper/) repository.

```python
# import modules
>>> import numpy as np 
>>> import jax.numpy as jnp
>>> from agginc import agginc, human_readable_dict # jax version
>>> # from agginc.np import agginc

# generate data for goodness-of-fit test
>>> perturbation = 0.5
>>> rs = jnp.random.RandomState(0)
>>> X = rs.gamma(5 + perturbation, 5, (500, 1))
>>> score_gamma = lambda x, k, theta : (k - 1) / x - 1 / theta
>>> score_X = score_gamma(X, 5, 5)
>>> X = jnp.array(X)
>>> score_X = jnp.array(score_X)

# run KSDAggInc test
>>> output = agginc("ksd", X, score_X)
>>> output
Array(1, dtype=int32)
>>> output.item()
1
>>> output, dictionary = agginc("ksd", X, score_X, return_dictionary=True)
>>> output
Array(1, dtype=int32)
>>> dictionary
>>> human_readable_dict(dictionary)
{'KSDAggInc test reject': True,
 'Single test 1': {'Bandwidth': 1.0,
  'KSD': 0.0005635482375510037,
  'KSD quantile': 0.0010665705194696784,
  'Kernel IMQ': True,
  'Reject': False,
  'p-value': 0.12974052131175995,
  'p-value threshold': 0.01596805267035961},
  ...
}
```

## Contact

If you have any issues running our tests, please do not hesitate to contact [Antonin Schrab](https://antoninschrab.github.io).

## Affiliations

Centre for Artificial Intelligence, Department of Computer Science, University College London

Gatsby Computational Neuroscience Unit, University College London

Inria London

## Bibtex

```
@inproceedings{schrab2022efficient,
  author    = {Antonin Schrab and Ilmun Kim and Benjamin Guedj and Arthur Gretton},
  title     = {Efficient Aggregated Kernel Tests using Incomplete \$U\$-statistics},
  booktitle = {Advances in Neural Information Processing Systems 35: Annual Conference
               on Neural Information Processing Systems 2022, NeurIPS 2022},
  editor    = {Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year      = {2022},
}
```

## License

MIT License (see [LICENSE.md](LICENSE.md)).
