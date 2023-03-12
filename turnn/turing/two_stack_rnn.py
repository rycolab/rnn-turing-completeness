from enum import IntEnum, unique
from itertools import product
from pprint import pprint

from sympy import zeros, eye, sympify, Symbol, Piecewise, Rational, Abs, Matrix

from turnn.base.symbol import Sym, BOT, EOS
from turnn.turing.pda import TwoStackPDA, Action

x = Symbol("x")
# This is the saturated sigmoid function
σ = Piecewise((Rational(0), x <= 0), (Rational(1), x >= 1), (Abs(x) <= sympify(1)))


@unique
class Index(IntEnum):
    """This class is used to index the hidden state of the Siegelmann RNN for two
    stacks by naming the individual dimensions of the hidden state with the role
    they play in simulating the two-stack PDA.
    """

    # commands
    STACK1 = 0
    STACK2 = 1
    BUFFER11 = 2
    BUFFER12 = 3
    BUFFER21 = 4
    BUFFER22 = 5

    PHASE1 = 6
    PHASE2 = 7
    PHASE3 = 8
    PHASE4 = 9

    STACK1_EMPTY = 10
    STACK1_ZERO = 11
    STACK1_ONE = 12
    STACK2_EMPTY = 13
    STACK2_ZERO = 14
    STACK2_ONE = 15

    # `CONF_γ1_γ2_a` corresponds to the configuration
    # where the top of the stack is `γ1`, `γ2` and the next symbol is `a`
    CONF_BOT_BOT_EOS = 16
    CONF_BOT_0_EOS = 17
    CONF_BOT_1_EOS = 18
    CONF_0_BOT_EOS = 19
    CONF_1_BOT_EOS = 20
    CONF_0_0_EOS = 21
    CONF_0_1_EOS = 22
    CONF_1_0_EOS = 23
    CONF_1_1_EOS = 24
    CONF_BOT_BOT_a = 25
    CONF_BOT_0_a = 26
    CONF_BOT_1_a = 27
    CONF_0_BOT_a = 28
    CONF_1_BOT_a = 29
    CONF_0_0_a = 30
    CONF_0_1_a = 31
    CONF_1_0_a = 32
    CONF_1_1_a = 33
    CONF_BOT_BOT_b = 34
    CONF_BOT_0_b = 35
    CONF_BOT_1_b = 36
    CONF_0_BOT_b = 37
    CONF_1_BOT_b = 38
    CONF_0_0_b = 39
    CONF_0_1_b = 40
    CONF_1_0_b = 41
    CONF_1_1_b = 42

    STACK1_PUSH_0 = 43
    STACK1_PUSH_1 = 44
    STACK1_POP_0 = 45
    STACK1_POP_1 = 46
    STACK1_NOOP = 47
    STACK2_PUSH_0 = 48
    STACK2_PUSH_1 = 49
    STACK2_POP_0 = 50
    STACK2_POP_1 = 51
    STACK2_NOOP = 52

    ACCEPT = 53


sym2idx = {
    Sym("a"): 0,
    Sym("b"): 1,
    EOS: 2,
}


def cantor_decode(x):
    stack = [BOT]
    if x.p == 0:
        return stack
    return stack + list(
        map(lambda x: Sym("0") if x == "1" else Sym("1"), reversed(list(str(x.p))))
    )


def conf_peek(x):  # noqa: C901
    if x == Index.CONF_BOT_BOT_EOS:
        print("(⊥, ⊥, EOS)")
    elif x == Index.CONF_BOT_0_EOS:
        print("(⊥, 0, EOS)")
    elif x == Index.CONF_BOT_1_EOS:
        print("(⊥, 1, EOS)")
    elif x == Index.CONF_0_BOT_EOS:
        print("(0, ⊥, EOS)")
    elif x == Index.CONF_1_BOT_EOS:
        print("(1, ⊥, EOS)")
    elif x == Index.CONF_BOT_BOT_a:
        print("(⊥, ⊥, a)")
    elif x == Index.CONF_BOT_BOT_b:
        print("(⊥, ⊥, b)")
    elif x == Index.CONF_BOT_0_a:
        print("(⊥, 0, a)")
    elif x == Index.CONF_BOT_0_b:
        print("(⊥, 0, b)")
    elif x == Index.CONF_BOT_1_a:
        print("(⊥, 1, a)")
    elif x == Index.CONF_BOT_1_b:
        print("(⊥, 1, b)")
    elif x == Index.CONF_0_BOT_a:
        print("(0, ⊥, a)")
    elif x == Index.CONF_0_BOT_b:
        print("(0, ⊥, b)")
    elif x == Index.CONF_1_BOT_a:
        print("(1, ⊥, a)")
    elif x == Index.CONF_1_BOT_b:
        print("(1, ⊥, b)")
    elif x == Index.CONF_0_0_a:
        print("(0, 0, a)")
    elif x == Index.CONF_0_0_b:
        print("(0, 0, b)")
    elif x == Index.CONF_0_1_a:
        print("(0, 1, a)")
    elif x == Index.CONF_0_1_b:
        print("(0, 1, b)")
    elif x == Index.CONF_1_0_a:
        print("(1, 0, a)")
    elif x == Index.CONF_1_0_b:
        print("(1, 0, b)")
    elif x == Index.CONF_1_1_a:
        print("(1, 1, a)")
    elif x == Index.CONF_1_1_b:
        print("(1, 1, b)")

    raise NotImplementedError


def conf2idx(γ1: Sym, γ2: Sym, sym: Sym):  # noqa: C901
    if (γ1, γ2, sym) == (BOT, BOT, EOS):
        return Index.CONF_BOT_BOT_EOS
    elif (γ1, γ2, sym) == (BOT, Sym("0"), EOS):
        return Index.CONF_BOT_0_EOS
    elif (γ1, γ2, sym) == (BOT, Sym("1"), EOS):
        return Index.CONF_BOT_1_EOS
    elif (γ1, γ2, sym) == (Sym("0"), BOT, EOS):
        return Index.CONF_0_BOT_EOS
    elif (γ1, γ2, sym) == (Sym("1"), BOT, EOS):
        return Index.CONF_1_BOT_EOS
    elif (γ1, γ2, sym) == (Sym("0"), Sym("0"), EOS):
        return Index.CONF_0_0_EOS
    elif (γ1, γ2, sym) == (Sym("0"), Sym("1"), EOS):
        return Index.CONF_0_1_EOS
    elif (γ1, γ2, sym) == (Sym("1"), Sym("0"), EOS):
        return Index.CONF_1_0_EOS
    elif (γ1, γ2, sym) == (Sym("1"), Sym("1"), EOS):
        return Index.CONF_1_1_EOS
    elif (γ1, γ2, sym) == (BOT, BOT, Sym("a")):
        return Index.CONF_BOT_BOT_a
    elif (γ1, γ2, sym) == (BOT, Sym("0"), Sym("a")):
        return Index.CONF_BOT_0_a
    elif (γ1, γ2, sym) == (BOT, Sym("1"), Sym("a")):
        return Index.CONF_BOT_1_a
    elif (γ1, γ2, sym) == (Sym("0"), BOT, Sym("a")):
        return Index.CONF_0_BOT_a
    elif (γ1, γ2, sym) == (Sym("1"), BOT, Sym("a")):
        return Index.CONF_1_BOT_a
    elif (γ1, γ2, sym) == (Sym("0"), Sym("0"), Sym("a")):
        return Index.CONF_0_0_a
    elif (γ1, γ2, sym) == (Sym("0"), Sym("1"), Sym("a")):
        return Index.CONF_0_1_a
    elif (γ1, γ2, sym) == (Sym("1"), Sym("0"), Sym("a")):
        return Index.CONF_1_0_a
    elif (γ1, γ2, sym) == (Sym("1"), Sym("1"), Sym("a")):
        return Index.CONF_1_1_a
    elif (γ1, γ2, sym) == (BOT, BOT, Sym("b")):
        return Index.CONF_BOT_BOT_b
    elif (γ1, γ2, sym) == (BOT, Sym("0"), Sym("b")):
        return Index.CONF_BOT_0_b
    elif (γ1, γ2, sym) == (BOT, Sym("1"), Sym("b")):
        return Index.CONF_BOT_1_b
    elif (γ1, γ2, sym) == (Sym("0"), BOT, Sym("b")):
        return Index.CONF_0_BOT_b
    elif (γ1, γ2, sym) == (Sym("1"), BOT, Sym("b")):
        return Index.CONF_1_BOT_b
    elif (γ1, γ2, sym) == (Sym("0"), Sym("0"), Sym("b")):
        return Index.CONF_0_0_b
    elif (γ1, γ2, sym) == (Sym("0"), Sym("1"), Sym("b")):
        return Index.CONF_0_1_b
    elif (γ1, γ2, sym) == (Sym("1"), Sym("0"), Sym("b")):
        return Index.CONF_1_0_b
    elif (γ1, γ2, sym) == (Sym("1"), Sym("1"), Sym("b")):
        return Index.CONF_1_1_b

    raise NotImplementedError


class TwoStackRNN:
    def __init__(self, pda: TwoStackPDA):
        self.pda = pda
        self.Σ = pda.Σ.union({EOS})  # In contrast to the globally normalized PDA,
        # the RNN works with an EOS symbol

        # The dimension of the hidden state
        self.D = len(Index)

        # Recurrence matrix U
        self.U = zeros(self.D, self.D, dtype=Rational(0))
        # Input matrix V
        self.V = zeros(self.D, len(self.Σ), dtype=Rational(0))
        # Bias vector b
        self.b = zeros(self.D, 1, dtype=Rational(0))

        self.make_U()
        self.make_b()
        self.make_V()

        # One-hot encoding of the input symbols
        self.one_hot = eye(len(self.Σ), dtype=Rational(0))

        # We start with an zero hidden state in phase 1
        # We start with an zero hidden state in phase 1
        self.h = zeros(self.D, 1, dtype=Rational(0))
        self.reset()

    def reset(self):
        """Sets the initial hidden state of the RNN to the zero vector
        indicating the first phase."""
        self.h = zeros(self.D, 1, dtype=Rational(0))
        self.h[Index.PHASE1] = Rational(1)

    def disp(self, h: Matrix):
        """Prints the content of the hidden state of the Siegelmann RNN, separated by
        the different components.

        Args:
            h (Matrix): The hidden state of the Siegelmann RNN.
        """
        print(f"Stack 1:\t{h[Index.STACK1]}")
        print(f"Buffer 11:\t{h[Index.BUFFER11]}")
        print(f"Buffer 21:\t{h[Index.BUFFER21]}")

        print(f"Stack 2:\t{h[Index.STACK2]}")
        print(f"Buffer 12:\t{h[Index.BUFFER12]}")
        print(f"Buffer 22:\t{h[Index.BUFFER22]}")

        print(f"Phase 1:\t{h[Index.PHASE1]}")
        print(f"Phase 2:\t{h[Index.PHASE2]}")
        print(f"Phase 3:\t{h[Index.PHASE3]}")
        print(f"Phase 4:\t{h[Index.PHASE4]}")

        print(f"1: STACK 1 - EMPTY:\t{h[Index.STACK1_EMPTY]}")
        print(f"1: STACK 1 - 0:\t{h[Index.STACK1_ZERO]}")
        print(f"1: STACK 1 - 1:\t{h[Index.STACK1_ONE]}")
        print(f"1: STACK 2 - EMPTY:\t{h[Index.STACK2_EMPTY]}")
        print(f"1: STACK 2 - 0:\t{h[Index.STACK2_ZERO]}")
        print(f"1: STACK 2 - 1:\t{h[Index.STACK2_ONE]}")

        print(f"2: (⊥, ⊥, EOS):\t{h[Index.CONF_BOT_BOT_EOS]}")
        print(f"2: (⊥, 0, EOS):\t{h[Index.CONF_BOT_0_EOS]}")
        print(f"2: (⊥, 1, EOS):\t{h[Index.CONF_BOT_1_EOS]}")
        print(f"2: (0, ⊥, EOS):\t{h[Index.CONF_0_BOT_EOS]}")
        print(f"2: (1, ⊥, EOS):\t{h[Index.CONF_1_BOT_EOS]}")
        print(f"2: (0, 0, EOS):\t{h[Index.CONF_0_0_EOS]}")
        print(f"2: (0, 1, EOS):\t{h[Index.CONF_0_1_EOS]}")
        print(f"2: (1, 0, EOS):\t{h[Index.CONF_1_0_EOS]}")
        print(f"2: (1, 1, EOS):\t{h[Index.CONF_1_1_EOS]}")
        print(f"2: (⊥, ⊥, a):\t{h[Index.CONF_BOT_BOT_a]}")
        print(f"2: (⊥, 0, a):\t{h[Index.CONF_BOT_0_a]}")
        print(f"2: (⊥, 1, a):\t{h[Index.CONF_BOT_1_a]}")
        print(f"2: (0, ⊥, a):\t{h[Index.CONF_0_BOT_a]}")
        print(f"2: (1, ⊥, a):\t{h[Index.CONF_1_BOT_a]}")
        print(f"2: (0, 0, a):\t{h[Index.CONF_0_0_a]}")
        print(f"2: (0, 1, a):\t{h[Index.CONF_0_1_a]}")
        print(f"2: (1, 0, a):\t{h[Index.CONF_1_0_a]}")
        print(f"2: (1, 1, a):\t{h[Index.CONF_1_1_a]}")
        print(f"2: (⊥, ⊥, b):\t{h[Index.CONF_BOT_BOT_b]}")
        print(f"2: (⊥, 0, b):\t{h[Index.CONF_BOT_0_b]}")
        print(f"2: (⊥, 1, b):\t{h[Index.CONF_BOT_1_b]}")
        print(f"2: (0, ⊥, b):\t{h[Index.CONF_0_BOT_b]}")
        print(f"2: (1, ⊥, b):\t{h[Index.CONF_1_BOT_b]}")
        print(f"2: (0, 0, b):\t{h[Index.CONF_0_0_b]}")
        print(f"2: (0, 1, b):\t{h[Index.CONF_0_1_b]}")
        print(f"2: (1, 0, b):\t{h[Index.CONF_1_0_b]}")
        print(f"2: (1, 1, b):\t{h[Index.CONF_1_1_b]}")

        print(f"3: STACK 1 Push 0:\t{h[Index.STACK1_PUSH_0]}")
        print(f"3: STACK 1 Push 1:\t{h[Index.STACK1_PUSH_1]}")
        print(f"3: STACK 1 Pop 0:\t{h[Index.STACK1_POP_0]}")
        print(f"3: STACK 1 Pop 1:\t{h[Index.STACK1_POP_1]}")

        print(f"3: STACK 2 Push 0:\t{h[Index.STACK2_PUSH_0]}")
        print(f"3: STACK 2 Push 1:\t{h[Index.STACK2_PUSH_1]}")
        print(f"3: STACK 2 Pop 0:\t{h[Index.STACK2_POP_0]}")
        print(f"3: STACK 2 Pop 1:\t{h[Index.STACK2_POP_1]}")

        print()

    def make_buffer_transitions(self):
        # This submatrix encodes the copying of the stack contents between
        # the three different stacks simulated by the RNN in the hidden state
        # while it moves through the 4 phases of the hidden state update.
        # Each of the stacks is simply a dimension in the hidden state.
        # Each of the three stacks is copied to the next one,
        # and the last one is zeroed out.
        self.U[Index.STACK1, Index.STACK1] = Rational(0)  # delete
        self.U[Index.BUFFER11, Index.STACK1] = Rational(1)  # copy over to the next one
        self.U[Index.BUFFER11, Index.BUFFER11] = Rational(0)
        self.U[Index.BUFFER21, Index.BUFFER11] = Rational(1)
        self.U[Index.BUFFER21, Index.BUFFER21] = Rational(0)

        self.U[Index.STACK2, Index.STACK2] = Rational(0)  # delete
        self.U[Index.BUFFER12, Index.STACK2] = Rational(1)  # copy over to the next one
        self.U[Index.BUFFER12, Index.BUFFER12] = Rational(0)
        self.U[Index.BUFFER22, Index.BUFFER12] = Rational(1)
        self.U[Index.BUFFER22, Index.BUFFER22] = Rational(0)

    def make_phase_transitions(self):
        # These two blocks encode the state---it's a one-hot encoding
        self.U[Index.PHASE1, Index.PHASE1] = Rational(0)
        self.U[Index.PHASE2, Index.PHASE2] = Rational(0)
        self.U[Index.PHASE3, Index.PHASE3] = Rational(0)
        self.U[Index.PHASE4, Index.PHASE4] = Rational(0)
        self.U[Index.PHASE2, Index.PHASE1] = Rational(1)
        self.U[Index.PHASE3, Index.PHASE2] = Rational(1)
        self.U[Index.PHASE4, Index.PHASE3] = Rational(1)
        self.U[Index.PHASE1, Index.PHASE4] = Rational(1)

    def make_stack_detector(self):
        # PHASE 1:
        # This detects an empty stack
        self.U[Index.STACK1_EMPTY, Index.STACK1] = Rational(-10)
        self.U[Index.STACK2_EMPTY, Index.STACK2] = Rational(-10)

        # This detects whether the top of the stack is a 0 or a 1
        self.U[Index.STACK1_ZERO, Index.STACK1] = Rational(-10)
        self.U[Index.STACK1_ONE, Index.STACK1] = Rational(10)
        self.U[Index.STACK2_ZERO, Index.STACK2] = Rational(-10)
        self.U[Index.STACK2_ONE, Index.STACK2] = Rational(10)

    def make_configuration_detector(self):
        # PHASE 2: This submatrix corresponds to the actions available given any
        # configuration of the stack, captured at EMPTY/PEEK locations.
        # This corresponds to the emmisions
        # These then get "intersected" with the symbol embeddings to select a or b
        self.U[Index.CONF_BOT_BOT_a, Index.STACK1_EMPTY] = Rational(1)
        self.U[Index.CONF_BOT_0_a, Index.STACK1_EMPTY] = Rational(1)
        self.U[Index.CONF_BOT_1_a, Index.STACK1_EMPTY] = Rational(1)
        self.U[Index.CONF_BOT_BOT_b, Index.STACK1_EMPTY] = Rational(1)
        self.U[Index.CONF_BOT_0_b, Index.STACK1_EMPTY] = Rational(1)
        self.U[Index.CONF_BOT_1_b, Index.STACK1_EMPTY] = Rational(1)
        self.U[Index.CONF_BOT_BOT_EOS, Index.STACK1_EMPTY] = Rational(1)
        self.U[Index.CONF_BOT_0_EOS, Index.STACK1_EMPTY] = Rational(1)
        self.U[Index.CONF_BOT_1_EOS, Index.STACK1_EMPTY] = Rational(1)
        self.U[Index.CONF_0_BOT_a, Index.STACK1_ZERO] = Rational(1)
        self.U[Index.CONF_0_0_a, Index.STACK1_ZERO] = Rational(1)
        self.U[Index.CONF_0_1_a, Index.STACK1_ZERO] = Rational(1)
        self.U[Index.CONF_0_BOT_b, Index.STACK1_ZERO] = Rational(1)
        self.U[Index.CONF_0_0_b, Index.STACK1_ZERO] = Rational(1)
        self.U[Index.CONF_0_1_b, Index.STACK1_ZERO] = Rational(1)
        self.U[Index.CONF_0_BOT_EOS, Index.STACK1_ZERO] = Rational(1)
        self.U[Index.CONF_0_0_EOS, Index.STACK1_ZERO] = Rational(1)
        self.U[Index.CONF_0_1_EOS, Index.STACK1_ZERO] = Rational(1)
        self.U[Index.CONF_1_BOT_a, Index.STACK1_ONE] = Rational(1)
        self.U[Index.CONF_1_0_a, Index.STACK1_ONE] = Rational(1)
        self.U[Index.CONF_1_1_a, Index.STACK1_ONE] = Rational(1)
        self.U[Index.CONF_1_BOT_b, Index.STACK1_ONE] = Rational(1)
        self.U[Index.CONF_1_0_b, Index.STACK1_ONE] = Rational(1)
        self.U[Index.CONF_1_1_b, Index.STACK1_ONE] = Rational(1)
        self.U[Index.CONF_1_BOT_EOS, Index.STACK1_ONE] = Rational(1)
        self.U[Index.CONF_1_0_EOS, Index.STACK1_ONE] = Rational(1)
        self.U[Index.CONF_1_1_EOS, Index.STACK1_ONE] = Rational(1)

        self.U[Index.CONF_BOT_BOT_a, Index.STACK2_EMPTY] = Rational(1)
        self.U[Index.CONF_BOT_BOT_b, Index.STACK2_EMPTY] = Rational(1)
        self.U[Index.CONF_0_BOT_b, Index.STACK2_EMPTY] = Rational(1)
        self.U[Index.CONF_0_BOT_a, Index.STACK2_EMPTY] = Rational(1)
        self.U[Index.CONF_1_BOT_a, Index.STACK2_EMPTY] = Rational(1)
        self.U[Index.CONF_1_BOT_b, Index.STACK2_EMPTY] = Rational(1)
        self.U[Index.CONF_BOT_0_a, Index.STACK2_ZERO] = Rational(1)
        self.U[Index.CONF_BOT_0_b, Index.STACK2_ZERO] = Rational(1)
        self.U[Index.CONF_0_0_a, Index.STACK2_ZERO] = Rational(1)
        self.U[Index.CONF_0_0_b, Index.STACK2_ZERO] = Rational(1)
        self.U[Index.CONF_1_0_a, Index.STACK2_ZERO] = Rational(1)
        self.U[Index.CONF_1_0_b, Index.STACK2_ZERO] = Rational(1)
        self.U[Index.CONF_BOT_1_a, Index.STACK2_ONE] = Rational(1)
        self.U[Index.CONF_BOT_1_b, Index.STACK2_ONE] = Rational(1)
        self.U[Index.CONF_0_1_a, Index.STACK2_ONE] = Rational(1)
        self.U[Index.CONF_0_1_b, Index.STACK2_ONE] = Rational(1)
        self.U[Index.CONF_1_1_a, Index.STACK2_ONE] = Rational(1)
        self.U[Index.CONF_1_1_b, Index.STACK2_ONE] = Rational(1)
        self.U[Index.CONF_BOT_BOT_EOS, Index.STACK2_EMPTY] = Rational(1)
        self.U[Index.CONF_0_BOT_EOS, Index.STACK2_EMPTY] = Rational(1)
        self.U[Index.CONF_1_BOT_EOS, Index.STACK2_EMPTY] = Rational(1)
        self.U[Index.CONF_BOT_0_EOS, Index.STACK2_ZERO] = Rational(1)
        self.U[Index.CONF_0_0_EOS, Index.STACK2_ZERO] = Rational(1)
        self.U[Index.CONF_1_1_EOS, Index.STACK2_ZERO] = Rational(1)
        self.U[Index.CONF_BOT_1_EOS, Index.STACK2_ONE] = Rational(1)
        self.U[Index.CONF_0_1_EOS, Index.STACK2_ONE] = Rational(1)
        self.U[Index.CONF_1_1_EOS, Index.STACK2_ONE] = Rational(1)

        # This bit ensures that we can't be empty and non-empty
        # It erases the effects of always setting the PEEK0 bit to 1
        self.U[Index.CONF_0_BOT_a, Index.STACK1_EMPTY] = Rational(-10)
        self.U[Index.CONF_0_0_a, Index.STACK1_EMPTY] = Rational(-10)
        self.U[Index.CONF_0_1_a, Index.STACK1_EMPTY] = Rational(-10)
        self.U[Index.CONF_0_BOT_b, Index.STACK1_EMPTY] = Rational(-10)
        self.U[Index.CONF_0_0_b, Index.STACK1_EMPTY] = Rational(-10)
        self.U[Index.CONF_0_1_b, Index.STACK1_EMPTY] = Rational(-10)
        self.U[Index.CONF_1_BOT_a, Index.STACK1_EMPTY] = Rational(-10)
        self.U[Index.CONF_1_0_a, Index.STACK1_EMPTY] = Rational(-10)
        self.U[Index.CONF_1_1_a, Index.STACK1_EMPTY] = Rational(-10)
        self.U[Index.CONF_1_BOT_b, Index.STACK1_EMPTY] = Rational(-10)
        self.U[Index.CONF_1_0_b, Index.STACK1_EMPTY] = Rational(-10)
        self.U[Index.CONF_1_1_b, Index.STACK1_EMPTY] = Rational(-10)

        self.U[Index.CONF_BOT_0_a, Index.STACK2_EMPTY] = Rational(-10)
        self.U[Index.CONF_0_0_a, Index.STACK2_EMPTY] = Rational(-10)
        self.U[Index.CONF_1_0_a, Index.STACK2_EMPTY] = Rational(-10)
        self.U[Index.CONF_BOT_0_b, Index.STACK2_EMPTY] = Rational(-10)
        self.U[Index.CONF_0_0_b, Index.STACK2_EMPTY] = Rational(-10)
        self.U[Index.CONF_1_0_b, Index.STACK2_EMPTY] = Rational(-10)
        self.U[Index.CONF_BOT_1_a, Index.STACK2_EMPTY] = Rational(-10)
        self.U[Index.CONF_0_1_a, Index.STACK2_EMPTY] = Rational(-10)
        self.U[Index.CONF_1_1_a, Index.STACK2_EMPTY] = Rational(-10)
        self.U[Index.CONF_BOT_1_b, Index.STACK2_EMPTY] = Rational(-10)
        self.U[Index.CONF_0_1_b, Index.STACK2_EMPTY] = Rational(-10)
        self.U[Index.CONF_1_1_b, Index.STACK2_EMPTY] = Rational(-10)

    def initialize_transition_function(self):
        # transition function
        for i in range(Index.STACK1_PUSH_0, Index.STACK1_NOOP + 1):
            for j in range(Index.CONF_BOT_BOT_EOS, Index.CONF_1_1_b + 1):
                # As soon as the configuration has been determined, reset
                # the information about what action should be taken,
                # so that only the correct action is considered (by overwriting
                # the -10's with 0's below).
                self.U[i, j] = Rational(-10)
        for i in range(Index.STACK2_PUSH_0, Index.STACK2_NOOP + 1):
            for j in range(Index.CONF_BOT_BOT_EOS, Index.CONF_1_1_b + 1):
                self.U[i, j] = Rational(-10)

    def make_action_detector(self):  # noqa: C901
        self.initialize_transition_function()

        # This is the transition matrix
        for sym in self.Σ:
            if sym == EOS:
                # If we read the EOS symbol, we don't do anything.
                # We set those manually, since EOS-transitions are not in the PDA.
                for action in [Index.STACK1_NOOP, Index.STACK2_NOOP]:
                    for γ_top_1, γ_top_2 in product(self.pda.Γ_1, self.pda.Γ_2):
                        self.U[action, conf2idx(γ_top_1, γ_top_2, EOS)] = Rational(0)
                continue

            for γ_top_1, γ_top_2 in self.pda.δ[sym]:
                (action_1, γ_new_1), (action_2, γ_new_2) = self.pda.δ[sym][
                    (γ_top_1, γ_top_2)
                ]

                conf = conf2idx(γ_top_1, γ_top_2, sym)

                if action_1 == Action.PUSH and γ_new_1 == Sym("0"):
                    self.U[Index.STACK1_PUSH_0, conf] = Rational(0)
                elif action_1 == Action.PUSH and γ_new_1 == Sym("1"):
                    self.U[Index.STACK1_PUSH_1, conf] = Rational(0)
                elif action_1 == Action.POP and γ_new_1 == Sym("0"):
                    self.U[Index.STACK1_POP_0, conf] = Rational(0)
                elif action_1 == Action.POP and γ_new_1 == Sym("1"):
                    self.U[Index.STACK1_POP_1, conf] = Rational(0)
                elif action_1 == Action.NOOP:
                    self.U[Index.STACK1_NOOP, conf] = Rational(0)
                else:
                    raise ValueError(
                        "Unknown action: action: {}, γ_new: {}.".format(
                            action_1, γ_new_1
                        )
                    )

                if action_2 == Action.PUSH and γ_new_2 == Sym("0"):
                    self.U[Index.STACK2_PUSH_0, conf] = Rational(0)
                elif action_2 == Action.PUSH and γ_new_2 == Sym("1"):
                    self.U[Index.STACK2_PUSH_1, conf] = Rational(0)
                elif action_2 == Action.POP and γ_new_2 == Sym("0"):
                    self.U[Index.STACK2_POP_0, conf] = Rational(0)
                elif action_2 == Action.POP and γ_new_2 == Sym("1"):
                    self.U[Index.STACK2_POP_1, conf] = Rational(0)
                elif action_2 == Action.NOOP:
                    self.U[Index.STACK2_NOOP, conf] = Rational(0)
                else:
                    raise ValueError(
                        "Unknown action: action: {}, γ_new: {}.".format(
                            action_2, γ_new_2
                        )
                    )

    def make_action_executor(self):
        # PHASE 3: we execute all possible actions
        # These two take whatever is on the stack so far (primarily it would be on
        # stack 1, but by phase 4, it has been moved to stack 3) and move it
        # down by dividing by 10.
        self.U[Index.STACK1_PUSH_0, Index.BUFFER21] = Rational("1/10")
        self.U[Index.STACK1_PUSH_1, Index.BUFFER21] = Rational("1/10")

        self.U[Index.STACK2_PUSH_0, Index.BUFFER22] = Rational("1/10")
        self.U[Index.STACK2_PUSH_1, Index.BUFFER22] = Rational("1/10")

        # These two take whatever is on the stack and move it up by multiplying by 10.
        self.U[Index.STACK1_POP_0, Index.BUFFER21] = Rational(10)
        self.U[Index.STACK1_POP_1, Index.BUFFER21] = Rational(10)

        self.U[Index.STACK2_POP_0, Index.BUFFER22] = Rational(10)
        self.U[Index.STACK2_POP_1, Index.BUFFER22] = Rational(10)

        # Copy the last buffer into the NOOP cells,
        # but only if none of the other actions
        #  are active (handled in `make_action_detector`).
        self.U[Index.STACK1_NOOP, Index.BUFFER21] = Rational(1)
        self.U[Index.STACK2_NOOP, Index.BUFFER22] = Rational(1)

    def make_action_committer(self):
        # PHASE 4: The final phase---commit the action from the individual
        # action cells to the stack.
        # Only one of the four entries PUSH_0, PUSH_1, POP_0, POP_1
        # will be "active" (non-zero), the rest will be 0.

        # This submatrix encodes the copying of the actions from the ``actions'' part
        # of the hidden state to the part of the hidden state corresponding to stack 1.

        self.U[Index.STACK1, Index.STACK1_PUSH_0] = Rational(1)
        self.U[Index.STACK1, Index.STACK1_PUSH_1] = Rational(1)
        self.U[Index.STACK2, Index.STACK2_PUSH_0] = Rational(1)
        self.U[Index.STACK2, Index.STACK2_PUSH_1] = Rational(1)

        self.U[Index.STACK1, Index.STACK1_POP_0] = Rational(1)
        self.U[Index.STACK1, Index.STACK1_POP_1] = Rational(1)
        self.U[Index.STACK2, Index.STACK2_POP_0] = Rational(1)
        self.U[Index.STACK2, Index.STACK2_POP_1] = Rational(1)

        self.U[Index.STACK1, Index.STACK1_NOOP] = Rational(1)
        self.U[Index.STACK2, Index.STACK2_NOOP] = Rational(1)

    def make_acceptor(self):
        # PHASE 5: acceptor
        # This is the final state of the RNN.
        # It accepts if the stack is empty (encoded in the computation component)
        # and the input is empty, i.e., the previous input was EOS.

        # Checks the emptiness of the computation component
        self.U[Index.ACCEPT, Index.STACK1_PUSH_0] = Rational(-10)
        self.U[Index.ACCEPT, Index.STACK1_PUSH_1] = Rational(-10)
        self.U[Index.ACCEPT, Index.STACK1_POP_0] = Rational(-10)
        self.U[Index.ACCEPT, Index.STACK1_POP_1] = Rational(-10)
        self.U[Index.ACCEPT, Index.STACK1_NOOP] = Rational(-10)
        self.U[Index.ACCEPT, Index.STACK2_PUSH_0] = Rational(-10)
        self.U[Index.ACCEPT, Index.STACK2_PUSH_1] = Rational(-10)
        self.U[Index.ACCEPT, Index.STACK2_POP_0] = Rational(-10)
        self.U[Index.ACCEPT, Index.STACK2_POP_1] = Rational(-10)
        self.U[Index.ACCEPT, Index.STACK2_NOOP] = Rational(-10)

        # Checks that the input was EOS
        self.U[Index.ACCEPT, Index.CONF_BOT_BOT_a] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_BOT_0_a] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_BOT_1_a] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_BOT_BOT_b] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_BOT_0_b] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_BOT_1_b] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_0_BOT_a] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_0_0_a] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_0_1_a] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_0_BOT_b] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_0_0_b] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_0_1_b] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_1_BOT_a] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_1_0_a] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_1_1_a] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_1_BOT_b] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_1_0_b] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_1_1_b] = Rational(-10)

    def make_U(self):
        """Constructs the recurrence matrix U of the Siegelmann RNN."""

        # ORCHESTRATION
        self.make_buffer_transitions()
        self.make_phase_transitions()

        # PHASE 1
        self.make_stack_detector()

        # PHASE 2
        self.make_configuration_detector()

        # PHASE 3
        self.make_action_detector()
        self.make_action_executor()

        # PHASE 4
        self.make_action_committer()
        self.make_acceptor()

    def make_V(self):
        """Constructs the emission matrix V of the Siegelmann RNN."""
        # Enables setting the hidden state to 1 at the indices
        # corresponding to all possible combinations of the stack configurations
        # (top of the stack) and the incomping input symbol.
        # For input symbol a
        self.V[Index.CONF_BOT_BOT_a, sym2idx[Sym("a")]] = Rational(1)
        self.V[Index.CONF_BOT_0_a, sym2idx[Sym("a")]] = Rational(1)
        self.V[Index.CONF_BOT_1_a, sym2idx[Sym("a")]] = Rational(1)
        self.V[Index.CONF_0_BOT_a, sym2idx[Sym("a")]] = Rational(1)
        self.V[Index.CONF_0_0_a, sym2idx[Sym("a")]] = Rational(1)
        self.V[Index.CONF_0_1_a, sym2idx[Sym("a")]] = Rational(1)
        self.V[Index.CONF_1_BOT_a, sym2idx[Sym("a")]] = Rational(1)
        self.V[Index.CONF_1_0_a, sym2idx[Sym("a")]] = Rational(1)
        self.V[Index.CONF_1_1_a, sym2idx[Sym("a")]] = Rational(1)
        # For input symbol b
        self.V[Index.CONF_BOT_BOT_b, sym2idx[Sym("b")]] = Rational(1)
        self.V[Index.CONF_BOT_0_b, sym2idx[Sym("b")]] = Rational(1)
        self.V[Index.CONF_BOT_1_b, sym2idx[Sym("b")]] = Rational(1)
        self.V[Index.CONF_0_BOT_b, sym2idx[Sym("b")]] = Rational(1)
        self.V[Index.CONF_0_0_b, sym2idx[Sym("b")]] = Rational(1)
        self.V[Index.CONF_0_1_b, sym2idx[Sym("b")]] = Rational(1)
        self.V[Index.CONF_1_BOT_b, sym2idx[Sym("b")]] = Rational(1)
        self.V[Index.CONF_1_0_b, sym2idx[Sym("b")]] = Rational(1)
        self.V[Index.CONF_1_1_b, sym2idx[Sym("b")]] = Rational(1)
        # For input symbol EOS
        self.V[Index.CONF_BOT_BOT_EOS, sym2idx[EOS]] = Rational(1)
        self.V[Index.CONF_BOT_0_EOS, sym2idx[EOS]] = Rational(1)
        self.V[Index.CONF_BOT_1_EOS, sym2idx[EOS]] = Rational(1)
        self.V[Index.CONF_0_BOT_EOS, sym2idx[EOS]] = Rational(1)
        self.V[Index.CONF_0_0_EOS, sym2idx[EOS]] = Rational(1)
        self.V[Index.CONF_0_1_EOS, sym2idx[EOS]] = Rational(1)
        self.V[Index.CONF_1_BOT_EOS, sym2idx[EOS]] = Rational(1)
        self.V[Index.CONF_1_0_EOS, sym2idx[EOS]] = Rational(1)
        self.V[Index.CONF_1_1_EOS, sym2idx[EOS]] = Rational(1)

    def make_b(self):
        """Constructs the bias vector b of the Siegelmann RNN."""

        self.b[Index.STACK1_EMPTY, 0] = Rational(1)
        self.b[Index.STACK2_EMPTY, 0] = Rational(1)
        self.b[Index.STACK1_ZERO, 0] = Rational(3)
        self.b[Index.STACK2_ZERO, 0] = Rational(3)
        self.b[Index.STACK1_ONE, 0] = Rational(-2)
        self.b[Index.STACK2_ONE, 0] = Rational(-2)

        # This provides the base of the quantities that get added to
        # the previous encoding of the stack after it is divided by 10.
        # Add the encoding of the new top of the stack
        self.b[Index.STACK1_PUSH_0, 0] = Rational("1/10")
        self.b[Index.STACK1_PUSH_1, 0] = Rational("3/10")

        self.b[Index.STACK2_PUSH_0, 0] = Rational("1/10")
        self.b[Index.STACK2_PUSH_1, 0] = Rational("3/10")

        # Remove the top encoding of the stack, after it has been multiplied by 10.
        self.b[Index.STACK1_POP_0, 0] = Rational(-1)
        self.b[Index.STACK1_POP_1, 0] = Rational(-3)

        self.b[Index.STACK2_POP_0, 0] = Rational(-1)
        self.b[Index.STACK2_POP_1, 0] = Rational(-3)

        self.b[Index.STACK1_NOOP, 0] = Rational(0)
        self.b[Index.STACK2_NOOP, 0] = Rational(0)

        # This zeroes out the configurations which are not active
        # even though some of the "triggering" cells in the hidden state are active.
        self.b[Index.CONF_BOT_BOT_a, 0] = Rational(-2)
        self.b[Index.CONF_BOT_0_a, 0] = Rational(-2)
        self.b[Index.CONF_BOT_1_a, 0] = Rational(-2)
        self.b[Index.CONF_BOT_BOT_b, 0] = Rational(-2)
        self.b[Index.CONF_BOT_0_b, 0] = Rational(-2)
        self.b[Index.CONF_BOT_1_b, 0] = Rational(-2)
        self.b[Index.CONF_BOT_BOT_EOS, 0] = Rational(-2)
        self.b[Index.CONF_BOT_0_EOS, 0] = Rational(-2)
        self.b[Index.CONF_BOT_1_EOS, 0] = Rational(-2)
        self.b[Index.CONF_0_BOT_a, 0] = Rational(-2)
        self.b[Index.CONF_0_0_a, 0] = Rational(-2)
        self.b[Index.CONF_0_1_a, 0] = Rational(-2)
        self.b[Index.CONF_0_BOT_b, 0] = Rational(-2)
        self.b[Index.CONF_0_0_b, 0] = Rational(-2)
        self.b[Index.CONF_0_1_b, 0] = Rational(-2)
        self.b[Index.CONF_0_BOT_EOS, 0] = Rational(-2)
        self.b[Index.CONF_0_0_EOS, 0] = Rational(-2)
        self.b[Index.CONF_0_1_EOS, 0] = Rational(-2)
        self.b[Index.CONF_1_BOT_a, 0] = Rational(-2)
        self.b[Index.CONF_1_0_a, 0] = Rational(-2)
        self.b[Index.CONF_1_1_a, 0] = Rational(-2)
        self.b[Index.CONF_1_BOT_b, 0] = Rational(-2)
        self.b[Index.CONF_1_0_b, 0] = Rational(-2)
        self.b[Index.CONF_1_1_b, 0] = Rational(-2)
        self.b[Index.CONF_1_BOT_EOS, 0] = Rational(-2)
        self.b[Index.CONF_1_0_EOS, 0] = Rational(-2)
        self.b[Index.CONF_1_1_EOS, 0] = Rational(-2)

        self.b[Index.ACCEPT, 0] = Rational(1)

    def __call__(self, a: Sym):
        """Performs a step of the Siegelmann RNN."""
        assert a in self.Σ

        # We have four stages in which the hidden state is updated
        # 1) Peek
        # 2) Combine
        # 3) Transition
        # 4) Consolidate

        def apply(_h):
            tmp = self.U * _h + self.V * self.one_hot.col(sym2idx[a]) + self.b

            self.disp(tmp)

            h2 = zeros(self.D, 1, dtype=Rational(0))

            for i in range(self.D):
                h2[i] = σ.subs(x, tmp[i])
            return h2

        h = self.h
        print("\n\n\n\n >>>> START:")
        self.disp(h)
        print(cantor_decode(h[Index.STACK1]), cantor_decode(h[Index.STACK2]))

        print("\n\n----------------------------\n>>>>> PHASE 1")
        h = apply(h)
        print(cantor_decode(h[Index.STACK1]), cantor_decode(h[Index.STACK2]))
        self.disp(h)
        print("\n\n----------------------------\n>>>>> PHASE 2")
        h = apply(h)
        print(cantor_decode(h[Index.STACK1]), cantor_decode(h[Index.STACK2]))
        self.disp(h)
        print("\n\n----------------------------\n>>>>> PHASE 3")
        h = apply(h)
        print(cantor_decode(h[Index.STACK1]), cantor_decode(h[Index.STACK2]))
        self.disp(h)
        print("\n\n----------------------------\n>>>>> PHASE 4")
        h = apply(h)
        print(cantor_decode(h[Index.STACK1]), cantor_decode(h[Index.STACK2]))
        self.disp(h)
        print(cantor_decode(h[Index.STACK1]), cantor_decode(h[Index.STACK2]))
        self.disp(h)
        self.h = h
        print("\n\n\n\n\n")

        return self.h, self.h[Index.ACCEPT]


def test(seed: int = 13):
    pda = TwoStackPDA(seed=seed)
    pda.step(Sym("a"))
    pda.step(Sym("a"))
    pda.step(Sym("b"))
    pda.step(Sym("a"))
    pda.step(Sym("b"))

    pprint(pda.δ)

    rnn = TwoStackRNN(pda)
    rnn(Sym("a"))
    rnn(Sym("a"))
    rnn(Sym("b"))
    rnn(Sym("a"))
    rnn(Sym("b"))
    rnn(EOS)

    print(pda.stacks)
    print(cantor_decode(rnn.h[Index.STACK1]), cantor_decode(rnn.h[Index.STACK2]))

    print(pda.stacks[0] == cantor_decode(rnn.h[Index.STACK1]))
    print(pda.stacks[1] == cantor_decode(rnn.h[Index.STACK2]))
