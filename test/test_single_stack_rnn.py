import random

from pytest import mark

from turnn.base.alphabet import to_alphabet
from turnn.base.symbol import EOS, BOT
from turnn.base.string import String
from turnn.turing.single_stack_rnn import (
    SingleStackRNN,
    Index,
    cantor_decode,
)
from turnn.turing.pda import SingleStackPDA


Σ = to_alphabet({"a", "b"})


@mark.parametrize("n_automata", [32])
@mark.parametrize("n_steps", [16, 32, 64])
def test_conversion(n_automata: int, n_steps: int):
    for seed in range(n_automata):
        y = String([random.choice(list(Σ)) for _ in range(n_steps)])

        pda = SingleStackPDA(seed=seed)
        rnn = SingleStackRNN(pda)

        for sym in y:
            pda.step(sym)
            rnn(sym)

        _, accept = rnn(EOS)

        assert pda.stack == cantor_decode(rnn.h[Index.STACK])
        assert pda.stack == [BOT] and accept or pda.stack != [BOT] and not accept
