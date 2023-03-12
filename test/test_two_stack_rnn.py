import random

from pytest import mark

from turnn.base.symbol import EOS, BOT
from turnn.base.alphabet import to_alphabet
from turnn.base.string import String
from turnn.turing.two_stack_rnn import TwoStackRNN, Index, cantor_decode
from turnn.turing.pda import TwoStackPDA


Σ = to_alphabet({"a", "b"})


@mark.parametrize("n_automata", [32])
@mark.parametrize("n_steps", [16, 32, 64])
def test_siegelmann(n_automata: int, n_steps: int):
    for seed in range(n_automata):
        y = String([random.choice(list(Σ)) for _ in range(n_steps)])

        pda = TwoStackPDA(seed=seed)
        rnn = TwoStackRNN(pda)

        for sym in y:
            pda.step(sym)
            rnn(sym)

        _, accept = rnn(EOS)

        assert pda.stacks[0] == cantor_decode(rnn.h[Index.STACK1])
        assert pda.stacks[1] == cantor_decode(rnn.h[Index.STACK2])

        assert (
            pda.stacks == [[BOT], [BOT]]
            and accept
            or pda.stacks != [[BOT], [BOT]]
            and not accept
        )
