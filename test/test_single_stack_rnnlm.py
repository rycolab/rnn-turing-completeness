import random

from turnn.base.alphabet import to_alphabet
from turnn.base.string import String
from turnn.turing.ppda import SingleStackPDA
from turnn.turing.single_stack_rnnlm import SingleStackRNN

Σ = to_alphabet({"a", "b"})


def test_conversion():
    for seed in range(20):
        y = String([random.choice(list(Σ)) for _ in range(16)])

        P = SingleStackPDA(seed=seed)
        R = SingleStackRNN(P)

        P_accept, P_logp = P(y)
        R_accept, R_logp = R(y)

        assert abs(P_logp - R_logp) < 1e-6

        assert P_accept and R_accept or not P_accept and not R_accept
