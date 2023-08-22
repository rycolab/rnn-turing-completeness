from enum import IntEnum, unique
from typing import Tuple, Union

from sympy import Abs, Matrix, Piecewise, Rational, Symbol, eye, log, sympify, zeros

from turnn.base.string import String
from turnn.base.symbol import BOT, EOS, Sym, ε
from turnn.turing.pda import Action, SingleStackPDA

x = Symbol("x")
σ = Piecewise((Rational(0), x <= 0), (Rational(1), x >= 1), (Abs(x) <= sympify(1)))


@unique
class Index(IntEnum):
    # (1) Data component
    STACK = 0
    BUFFER1 = 1
    BUFFER2 = 2

    # (2) Phase component
    PHASE1 = 3
    PHASE2 = 4
    PHASE3 = 5
    PHASE4 = 6

    # (3) Stack configuration component
    # Since we assume a single-state machine, this is also
    # the full configuration of the machine.
    STACK_EMPTY = 7
    STACK_ZERO = 8
    STACK_ONE = 9

    # (4) Stack and input configuration component
    # `CONF_γ_a` corresponds to the configuration
    # where the top of the stack is `γ` and the next symbol is `a`
    CONF_BOT_ε = 10
    CONF_BOT_a = 11
    CONF_BOT_b = 12
    CONF_BOT_EOS = 13
    CONF_0_ε = 14
    CONF_0_a = 15
    CONF_0_b = 16
    CONF_0_EOS = 17
    CONF_1_ε = 18
    CONF_1_a = 19
    CONF_1_b = 20
    CONF_1_EOS = 21

    # (5) Computation component
    PUSH_0 = 22
    PUSH_1 = 23
    POP_0 = 24
    POP_1 = 25
    NOOP = 26

    # (6) Acceptance component
    ACCEPT = 27


def cantor_decode(x):
    stack = [BOT]
    if x.p == 0:
        return stack
    return stack + list(
        map(lambda x: Sym("0") if x == "1" else Sym("1"), reversed(list(str(x.p))))
    )


def enc_peek(x):  # noqa: C901
    if x == Index.CONF_BOT_a:
        print("(⊥, a)")
    elif x == Index.CONF_BOT_b:
        print("(⊥, b)")
    elif x == Index.CONF_0_a:
        print("(0, a)")
    elif x == Index.CONF_0_b:
        print("(0, b)")
    elif x == Index.CONF_1_a:
        print("(1, a)")
    elif x == Index.CONF_1_b:
        print("(1, b)")
    elif x == Index.CONF_BOT_EOS:
        print("(⊥, EOS)")
    elif x == Index.CONF_0_EOS:
        print("(0, EOS)")
    elif x == Index.CONF_1_EOS:
        print("(1, EOS)")

    raise Exception


def conf2idx(γ: Sym, sym: Sym):  # noqa: C901
    if (sym, γ) == (ε, BOT):
        return Index.CONF_BOT_ε
    elif (sym, γ) == (Sym("a"), BOT):
        return Index.CONF_BOT_a
    elif (sym, γ) == (Sym("b"), BOT):
        return Index.CONF_BOT_b
    elif (sym, γ) == (EOS, BOT):
        return Index.CONF_BOT_EOS
    elif (sym, γ) == (ε, Sym("0")):
        return Index.CONF_0_ε
    elif (sym, γ) == (Sym("a"), Sym("0")):
        return Index.CONF_0_a
    elif (sym, γ) == (Sym("b"), Sym("0")):
        return Index.CONF_0_b
    elif (sym, γ) == (EOS, Sym("0")):
        return Index.CONF_0_EOS
    elif (sym, γ) == (ε, Sym("1")):
        return Index.CONF_1_ε
    elif (sym, γ) == (Sym("a"), Sym("1")):
        return Index.CONF_1_a
    elif (sym, γ) == (Sym("b"), Sym("1")):
        return Index.CONF_1_b
    elif (sym, γ) == (EOS, Sym("1")):
        return Index.CONF_1_EOS
    raise ValueError("Unknown configuration: ({}, {}).".format(γ, sym))


class SingleStackRNN:
    def __init__(self, pda: SingleStackPDA):
        self.pda = pda
        self.Σ = pda.Σ.union({EOS})  # In contrast to the globally normalized PDA,
        # the RNN works with an EOS symbol

        self.sym2idx = {sym: idx for idx, sym in enumerate(self.Σ)}

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

        # Emission matrix E
        self.E = zeros(len(self.Σ), self.D, dtype=Rational(0))
        self.make_E()

        # One-hot encoding of the input symbols
        self.one_hot = eye(len(self.Σ), dtype=Rational(0))

    def disp(self, h: Matrix):
        """Prints the content of the hidden state of the RNN, separated by
        the different components.

        Args:
            h (Matrix): The hidden state of the RNN.
        """
        print(f"Stack:\t{h[Index.STACK]}")
        print(f"Buffer 1:\t{h[Index.BUFFER1]}")
        print(f"Buffer 2:\t{h[Index.BUFFER2]}")

        print(f"Phase 1:\t{h[Index.PHASE1]}")
        print(f"Phase 2:\t{h[Index.PHASE2]}")
        print(f"Phase 3:\t{h[Index.PHASE3]}")
        print(f"Phase 4:\t{h[Index.PHASE4]}")

        print(f"1 Peek EMPTY:\t{h[Index.STACK_EMPTY]}")
        print(f"1 Peek 0:\t{h[Index.STACK_ZERO]}")
        print(f"1 Peek 1:\t{h[Index.STACK_ONE]}")

        print(f"2 (⊥, a):\t{h[Index.CONF_BOT_a]}")
        print(f"2 (⊥, b):\t{h[Index.CONF_BOT_b]}")
        print(f"2 (0, a):\t{h[Index.CONF_0_a]}")
        print(f"2 (0, b):\t{h[Index.CONF_0_b]}")
        print(f"2 (1, a):\t{h[Index.CONF_1_a]}")
        print(f"2 (1, b):\t{h[Index.CONF_1_b]}")
        print(f"2 (BOT, EOS):\t{h[Index.CONF_BOT_EOS]}")
        print(f"2 (0, EOS):\t{h[Index.CONF_0_EOS]}")
        print(f"2 (1, EOS):\t{h[Index.CONF_1_EOS]}")

        print(f"3 Push 0:\t{h[Index.PUSH_0]}")
        print(f"3 Push 1:\t{h[Index.PUSH_1]}")
        print(f"3 Pop 0:\t{h[Index.POP_0]}")
        print(f"3 Pop 1:\t{h[Index.POP_1]}")
        print(f"3 NOOP:\t{h[Index.NOOP]}")

        print(f"4 ACCEPT:\t{h[Index.ACCEPT]}")

        print()

    def make_stack_transitions(self):
        # This submatrix encodes the copying of the stack contents between
        # the three different stacks simulated by the RNN in the hidden state
        # while it moves through the 4 phases of the hidden state update.
        # Each of the stacks is simply a dimension in the hidden state.
        # Each of the three stacks is copied to the next one,
        # and the last one is zeroed out.
        self.U[Index.STACK, Index.STACK] = Rational(0)  # delete
        self.U[Index.BUFFER1, Index.STACK] = Rational(1)  # copy over to the next one
        self.U[Index.BUFFER1, Index.BUFFER1] = Rational(0)
        self.U[Index.BUFFER2, Index.BUFFER1] = Rational(1)
        self.U[Index.BUFFER2, Index.BUFFER2] = Rational(0)

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
        self.U[Index.STACK_EMPTY, Index.STACK] = Rational(-10)
        # This detects whether the top of the stack is a 0 or a 1
        self.U[Index.STACK_ZERO, Index.STACK] = Rational(-10)
        self.U[Index.STACK_ONE, Index.STACK] = Rational(10)

    def make_configuration_detector(self):
        # PHASE 2: This submatrix corresponds to the actions available given any
        # configuration of the stack, captured at EMPTY/PEEK locations.
        # This corresponds to the emmisions
        # These then get "intersected" with the symbol embeddings to select a or b
        self.U[Index.CONF_BOT_a, Index.STACK_EMPTY] = Rational(1)
        self.U[Index.CONF_BOT_b, Index.STACK_EMPTY] = Rational(1)
        self.U[Index.CONF_BOT_EOS, Index.STACK_EMPTY] = Rational(1)
        # E.g., given that we have peeked 0, we can either read in an ``a'' or a ``b''
        self.U[Index.CONF_0_a, Index.STACK_ZERO] = Rational(1)
        self.U[Index.CONF_0_b, Index.STACK_ZERO] = Rational(1)
        self.U[Index.CONF_0_EOS, Index.STACK_ZERO] = Rational(1)
        self.U[Index.CONF_1_a, Index.STACK_ONE] = Rational(1)
        self.U[Index.CONF_1_b, Index.STACK_ONE] = Rational(1)
        self.U[Index.CONF_1_EOS, Index.STACK_ONE] = Rational(1)

        # This bit ensures that we can't be empty and non-empty
        # It erases the effects of always setting the PEEK0 bit to 1
        self.U[Index.STACK_ZERO, Index.STACK_EMPTY] = Rational(-3)
        self.U[Index.CONF_0_a, Index.STACK_EMPTY] = Rational(-1)
        self.U[Index.CONF_0_b, Index.STACK_EMPTY] = Rational(-1)
        self.U[Index.CONF_0_EOS, Index.STACK_EMPTY] = Rational(-1)
        # self.U[Index.CONF_1_a, Index.STACK_EMPTY] = Rational(-10)
        # self.U[Index.CONF_1_b, Index.STACK_EMPTY] = Rational(-10)
        # self.U[Index.CONF_1_EOS, Index.STACK_EMPTY] = Rational(-10)

    def initialize_transition_function(self):
        # transition function
        for i in range(Index.PUSH_0, Index.NOOP + 1):
            for j in range(Index.CONF_BOT_a, Index.CONF_1_EOS + 1):
                # As soon as the configuration has been determined, reset
                # the information about what action should be taken,
                # so that only the correct action is considered (by overwriting
                # the -10's with 0's below).
                self.U[i, j] = Rational(-10)

    def make_action_detector(self):  # noqa: C901
        self.initialize_transition_function()

        # This is the transition matrix
        for sym in self.Σ:
            if sym == EOS:
                # If we read the EOS symbol, we don't do anything.
                # We set those manually, since EOS-transitions are not in the PDA.
                self.U[Index.NOOP, conf2idx(BOT, EOS)] = Rational(0)
                self.U[Index.NOOP, conf2idx(Sym("0"), EOS)] = Rational(0)
                self.U[Index.NOOP, conf2idx(Sym("1"), EOS)] = Rational(0)
                continue

            for γ in self.pda.δ[sym]:
                action, γʼ = self.pda.δ[sym][γ][0]
                conf = conf2idx(γ, sym)

                if action == Action.PUSH and γʼ == Sym("0"):
                    # If the PDA stack top is γ1 and sym is read, simulate PUSHing 0
                    # According to the PDA: if we were in the configuration encoded by
                    # conf, and we read sym, we would push 0 onto the stack.
                    self.U[Index.PUSH_0, conf] = Rational(0)
                elif action == Action.PUSH and γʼ == Sym("1"):
                    # If the PDA stack top is γ1 and sym is read, simulate PUSHing 1
                    self.U[Index.PUSH_1, conf] = Rational(0)
                elif action == Action.POP and γʼ == Sym("0"):
                    # If the PDA stack top is γ1 and sym is read,
                    # simulate POPping γ1 (0)
                    self.U[Index.POP_0, conf] = Rational(0)
                elif action == Action.POP and γʼ == Sym("1"):
                    # If the PDA stack top is γ1 and sym is read,
                    # simulate POPping γ1 (1)
                    self.U[Index.POP_1, conf] = Rational(0)
                elif action == Action.NOOP:
                    self.U[Index.NOOP, conf] = Rational(0)
                else:
                    raise ValueError(
                        "Unknown action: action: {}, γʼ: {}.".format(action, γʼ)
                    )

    def make_action_executor(self):
        # PHASE 3: we execute all possible actions
        # These two take whatever is on the stack so far (primarily it would be on
        # stack 1, but by phase 4, it has been moved to stack 3) and move it
        # down by dividing by 10.
        self.U[Index.PUSH_0, Index.BUFFER2] = Rational("1/10")
        self.U[Index.PUSH_1, Index.BUFFER2] = Rational("1/10")
        # These two take whatever is on the stack and move it up by multiplying by 10.
        self.U[Index.POP_0, Index.BUFFER2] = Rational(10)
        self.U[Index.POP_1, Index.BUFFER2] = Rational(10)

        # Copy the last buffer into the NOOP cell, but only if none of the other actions
        # are active (handled in `make_action_detector`).
        self.U[Index.NOOP, Index.BUFFER2] = Rational(1)

    def make_action_committer(self):
        # PHASE 4: The final phase---commit the action from the individual
        # action cells to the stack.
        # Only one of the four entries PUSH_0, PUSH_1, POP_0, POP_1
        # will be "active" (non-zero), the rest will be 0.

        # This submatrix encodes the copying of the actions from the ``actions'' part
        # of the hidden state to the part of the hidden state corresponding to stack 1.
        self.U[Index.STACK, Index.PUSH_0] = Rational(1)
        self.U[Index.STACK, Index.PUSH_1] = Rational(1)
        self.U[Index.STACK, Index.POP_0] = Rational(1)
        self.U[Index.STACK, Index.POP_1] = Rational(1)
        self.U[Index.STACK, Index.NOOP] = Rational(1)

    def make_acceptor(self):
        # PHASE 5: acceptor
        # This is the final state of the RNN.
        # It accepts if the stack is empty (encoded in the computation component)
        # and the input is empty, i.e., the previous input was EOS.

        # Checks the emptiness of the computation component
        self.U[Index.ACCEPT, Index.PUSH_0] = Rational(-10)
        self.U[Index.ACCEPT, Index.PUSH_1] = Rational(-10)
        self.U[Index.ACCEPT, Index.POP_0] = Rational(-10)
        self.U[Index.ACCEPT, Index.POP_1] = Rational(-10)
        self.U[Index.ACCEPT, Index.NOOP] = Rational(-10)

        # Checks that the input was EOS
        self.U[Index.ACCEPT, Index.CONF_BOT_a] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_BOT_b] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_0_a] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_0_b] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_1_a] = Rational(-10)
        self.U[Index.ACCEPT, Index.CONF_1_b] = Rational(-10)

    def make_U(self):
        """Constructs the recurrence matrix U of the RNN."""

        # ORCHESTRATION
        self.make_stack_transitions()
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
        """Constructs the emission matrix V of the RNN."""
        # Enables setting the hidden state to 1 at the indices
        # corresponding to all possible combinations of the stack configurations
        # (top of the stack) and the incomping input symbol.
        # For input symbol a
        self.V[Index.CONF_BOT_a, self.sym2idx[Sym("a")]] = Rational(1)
        self.V[Index.CONF_0_a, self.sym2idx[Sym("a")]] = Rational(1)
        self.V[Index.CONF_1_a, self.sym2idx[Sym("a")]] = Rational(1)
        # For input symbol b
        self.V[Index.CONF_BOT_b, self.sym2idx[Sym("b")]] = Rational(1)
        self.V[Index.CONF_0_b, self.sym2idx[Sym("b")]] = Rational(1)
        self.V[Index.CONF_1_b, self.sym2idx[Sym("b")]] = Rational(1)
        # For input symbol EOS
        self.V[Index.CONF_BOT_EOS, self.sym2idx[EOS]] = Rational(1)
        self.V[Index.CONF_0_EOS, self.sym2idx[EOS]] = Rational(1)
        self.V[Index.CONF_1_EOS, self.sym2idx[EOS]] = Rational(1)

    def make_b(self):
        """Constructs the bias vector b of the RNN."""
        self.b[Index.STACK_EMPTY, 0] = Rational(1)
        # If the top of the stack is a 0, the encoding will be 0.1...
        # This means that -10 * stack encoding (self.U[Index.STACK_ZERO, Index.STACK1])
        # will be at least -2.
        # We therefore map it so a value >= 1.
        # Otherwise, the encoding will be 0.3... and this, together with the
        # multiplication and addition will not be >= 1.
        self.b[Index.STACK_ZERO, 0] = Rational(3)
        # If the top of the stack is a 1, the encoding will be 0.3...
        # This means that 10 * stack encoding (self.U[Index.STACK_ONE, Index.STACK1])
        # will be at least 3.
        # We therefore map it so a value >= 1.
        # Otherwise, the encoding will be 0.1... and this, together with the
        # multiplication and subtraction will not be >= 1.
        self.b[Index.STACK_ONE, 0] = Rational(-2)

        # This provides the base of the quantities that get added to
        # the previous encoding of the stack after it is divided by 10.
        # Add the encoding of the new top of the stack
        self.b[Index.PUSH_0, 0] = Rational("1/10")
        self.b[Index.PUSH_1, 0] = Rational("3/10")
        # Remove the top encoding of the stack, after it has been multiplied by 10.
        self.b[Index.POP_0, 0] = Rational(-1)
        self.b[Index.POP_1, 0] = Rational(-3)

        self.b[Index.NOOP, 0] = Rational(0)

        # This zeroes out the configurations which are not active
        # even though some of the "triggering" cells in the hidden state are active.
        self.b[Index.CONF_BOT_a, 0] = Rational(-1)
        self.b[Index.CONF_BOT_b, 0] = Rational(-1)
        self.b[Index.CONF_BOT_EOS, 0] = Rational(-1)
        self.b[Index.CONF_0_a, 0] = Rational(-1)
        self.b[Index.CONF_0_b, 0] = Rational(-1)
        self.b[Index.CONF_0_EOS, 0] = Rational(-1)
        self.b[Index.CONF_1_a, 0] = Rational(-1)
        self.b[Index.CONF_1_b, 0] = Rational(-1)
        self.b[Index.CONF_1_EOS, 0] = Rational(-1)

        self.b[Index.ACCEPT, 0] = Rational(1)

    def make_E(self):
        for a in self.Σ - {EOS}:
            d = self.sym2idx[a]
            self.E[d, Index.STACK_EMPTY] = log(self.pda.δ[a][BOT][1])
            self.E[d, Index.STACK_ZERO] = log(self.pda.δ[a][Sym("0")][1])
            self.E[d, Index.STACK_ONE] = log(self.pda.δ[a][Sym("1")][1])

    def initial_hidden_state(self) -> Matrix:
        """Sets the initial hidden state of the RNN to the zero vector
        indicating the first phase."""
        # We start with an zero hidden state in phase 1
        h = zeros(self.D, 1, dtype=Rational(0))
        h[Index.PHASE1] = Rational(1)
        return h

    def accept(self, y: Union[str, String]) -> Tuple[bool, float]:
        """Checks whether the RNN accepts the input string `y`.

        Args:
            y (String): The input string.

        Returns:
            bool: True if the RNN accepts the input string, False otherwise.
        """
        if isinstance(y, str):
            y = String(y)
        y.y += [EOS]

        h, logp = self.initial_hidden_state(), Rational(0)
        for a in y:
            print(f"sym = {a}")
            h, _acc, _logp = self.step(h, a)
            print(f"logp = {_logp}")
            logp += _logp

        return h[Index.ACCEPT] == Rational(1), logp

    def __call__(self, y: Union[str, String]) -> Tuple[bool, float]:
        return self.accept(y)

    def step(self, h: Matrix, a: Sym) -> Tuple[Matrix, Rational, Rational]:
        """Performs a step of the RNN."""
        assert a in self.Σ

        # We have four stages in which the hidden state is updated
        # 1) Peek
        # 2) Combine
        # 3) Transition
        # 4) Consolidate

        def apply(_h):
            z = self.U * _h + self.V * self.one_hot.col(self.sym2idx[a]) + self.b
            hʼ = zeros(self.D, 1, dtype=Rational(0))

            for i in range(self.D):
                hʼ[i] = σ.subs(x, z[i])
            return hʼ

        # print("\n\n\n\n\n\n >>>> START:")
        # self.disp(h)
        # print(cantor_decode(h[Index.STACK]))

        print("\n\n\n\n----------------------------\n>>>>> PHASE 1")
        h = apply(h)
        # print(cantor_decode(h[Index.STACK]))
        self.disp(h)
        print("\n\n\n\n----------------------------\n>>>>> PHASE 2")
        h = apply(h)
        # print(cantor_decode(h[Index.STACK]))
        self.disp(h)
        logits = self.E * h
        # print("\n\n\n\n----------------------------\n>>>>> PHASE 3")
        h = apply(h)
        # print(cantor_decode(h[Index.STACK]))
        # self.disp(h)
        # print("\n\n\n\n----------------------------\n>>>>> PHASE 4")
        h = apply(h)
        # print(cantor_decode(h[Index.STACK]))
        # self.disp(h)
        # print(cantor_decode(h[Index.STACK]))
        # self.disp(h)

        return h, h[Index.ACCEPT], logits[self.sym2idx[a]]
