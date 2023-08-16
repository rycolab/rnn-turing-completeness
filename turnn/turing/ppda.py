import random
from enum import IntEnum, unique
from itertools import product
from math import log
from typing import List, Tuple, Union

import numpy as np

from turnn.base.string import String
from turnn.base.symbol import BOT, Sym, ε


@unique
class Action(IntEnum):
    PUSH = 0
    POP = 1
    NOOP = 2


class SingleStackPDA:
    def __init__(
        self,
        # Σ={Sym("a"), Sym("b"), ε},
        Σ={Sym("a"), Sym("b")},
        Γ={BOT, Sym("0"), Sym("1")},
        seed: int = 42,
        randomize: bool = True,
    ):
        self.seed = seed

        self.Σ = Σ
        self.Γ = Γ

        # δ: Σ × Γ → (({PUSH, POP, NOOP} × Γ) × R)
        self.δ = {sym: {γ: {} for γ in self.Γ} for sym in self.Σ}
        if randomize:
            self._random_δ()

    def step(self, stack: List[Sym], a: Sym) -> Tuple[List[Sym], float]:
        assert a in self.Σ

        γ = stack[-1]
        action, γʼ = self.δ[a][γ][0]

        if action == Action.PUSH:
            stack.append(γʼ)
        elif action == Action.POP:
            stack.pop()
        elif action == Action.NOOP:
            pass
        else:
            raise Exception

        return stack, self.δ[a][γ][1]

    @property
    def probabilistic(self):
        d = {γ: 0 for γ in self.Γ}
        for a, γ in product(self.Σ, self.Γ):
            d[γ] += self.δ[a][γ][1]
        return all(abs(d[γ] - 1) < 1e-6 for γ in self.Γ)

    def accept(self, y: Union[str, String]) -> Tuple[bool, float]:
        if isinstance(y, str):
            y = String(y)

        stack = [BOT]
        logp = 0

        for a in y:
            stack, p = self.step(stack, a)
            logp += log(p)
            print(f"logp: {log(p)}")

        return stack == [BOT], logp

    def __call__(self, y: Union[str, String]) -> Tuple[bool, float]:
        return self.accept(y)

    def _random_δ(self):
        rng = np.random.default_rng(self.seed)

        pushes = list((Action.PUSH, γ) for γ in (self.Γ - {BOT}))

        for γ in self.Γ:
            # The possible actions have to form a probability distribution for every γ.
            α = rng.dirichlet(np.ones(len(self.Σ)))
            for ii, a in enumerate(self.Σ):
                if γ == BOT:
                    flip = rng.integers(0, 1, endpoint=True)
                    if flip == 0:
                        self.δ[a][γ] = ((Action.NOOP, γ), α[ii])
                    else:
                        self.δ[a][γ] = (tuple(rng.choice(pushes)), α[ii])
                else:
                    flip = rng.integers(0, 2, endpoint=True)
                    if flip == 0:
                        self.δ[a][γ] = ((Action.NOOP, γ), α[ii])
                    elif flip == 1:
                        self.δ[a][γ] = ((Action.POP, γ), α[ii])
                    else:
                        self.δ[a][γ] = (tuple(rng.choice(pushes)), α[ii])


class TwoStackPDA:
    def __init__(
        self,
        Σ={Sym("a"), Sym("b")},
        Γ_1={BOT, Sym("0"), Sym("1")},
        Γ_2={BOT, Sym("0"), Sym("1")},
        seed: int = 42,
    ):
        self.seed = seed

        self.Σ = Σ
        self.Γ_1 = Γ_1
        self.Γ_2 = Γ_2

        self.stacks = [[BOT], [BOT]]
        self.δ = {}
        self._random_δ()

    def _execute_action(self, stack, action, γ_new):
        if action == Action.PUSH:
            stack.append(γ_new)
        elif action == Action.POP:
            stack.pop()
        elif action == Action.NOOP:
            pass
        else:
            raise Exception

    def step(self, y):
        assert y in self.Σ

        γ_top_1, γ_top_2 = self.stacks[0][-1], self.stacks[1][-1]
        (action_1, γ_new_1), (action_2, γ_new_2) = self.δ[y][(γ_top_1, γ_top_2)]

        self._execute_action(self.stacks[0], action_1, γ_new_1)
        self._execute_action(self.stacks[1], action_2, γ_new_2)

    def accept(self, y):
        for sym in y:
            self.step(sym)

        return self.stacks[0] == [BOT] and self.stacks[1] == [BOT]

    def _get_action(self, γ_top: Sym, pushes: List[Tuple[Action, Sym]]):
        if γ_top == BOT:
            flip = random.randint(0, 1)
            if flip == 0:
                action, γ_new = (Action.NOOP, γ_top)
            else:
                action, γ_new = random.choice(pushes)
        else:
            flip = random.randint(0, 2)
            if flip == 0:
                action, γ_new = (Action.NOOP, γ_top)
            elif flip == 1:
                action, γ_new = (Action.POP, γ_top)
            else:
                action, γ_new = random.choice(pushes)

        return action, γ_new

    def _random_δ(self):
        random.seed(self.seed)

        pushes = [
            [(Action.PUSH, γ) for γ in Γ if γ != BOT] for Γ in [self.Γ_1, self.Γ_2]
        ]

        for a in self.Σ:
            if a not in self.δ:
                self.δ[a] = {}

            for γ_1, γ_2 in product(self.Γ_1, self.Γ_2):
                action_1, γ_new_1 = self._get_action(γ_1, pushes[0])
                action_2, γ_new_2 = self._get_action(γ_2, pushes[1])

                self.δ[a][(γ_1, γ_2)] = (action_1, γ_new_1), (action_2, γ_new_2)
