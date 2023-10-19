Code accompanying the EMNLP 2023 publication "On the Representational Capacity of Recurrent Neural Language Models".

This repository contains a library for proving the Turing completeness of recurrent neural network language models. 
Specifically, it includes implementations of weighted single-stack and two-stack pushdown automata encoded in recurrent neural nets (RNNs). 
Since two-stack pushdown automata are equivalent to Turing machines, this shows by construction that RNNs are Turing complete, and can even simulate certain kinds of probabilistic Turing machines.

Implementation programmed by Ryan Cotterell and Anej Svete.
The original inspiration for the "proof by code" is [Siegelmann and Sontag (1995)](https://binds.cs.umass.edu/papers/1995_Siegelmann_JComSysSci.pdf). 

## Getting started with the code

Clone the repository:

```bash
$ git clone https://github.com/rycolab/rnn-turing-completeness.git
$ cd rnn-turing-completeness
$ pip install -e .
```

At this point it may be beneficial to create a new [Python virtual environment](https://docs.python.org/3.8/tutorial/venv.html). 
There are multiple solutions for this step, including [Miniconda](https://docs.conda.io/en/latest/miniconda.html). 
We aim at Python 3.10 version and above.

Then you install the package _in editable mode_:
```bash
$ pip install -e .
```

We use [black](https://github.com/psf/black) and [flake8](https://flake8.pycqa.org/en/latest/) to lint the code, [pytype](https://github.com/google/pytype) to check whether the types agree, and [pytest](https://docs.pytest.org) to unit test the code.

To **unit-test** the code, run:
```
pytest .
```
