import random
from itertools import product
from enum import IntEnum, unique
from typing import Tuple, List

from turnn.base.symbol import BOT, Sym
from turnn.base.string import String


@unique
class Action(IntEnum):
    PUSH = 0
    POP = 1
    NOOP = 2


class SingleStackPDA:
    def __init__(
        self,
        Σ={Sym("a"), Sym("b")},
        Γ={BOT, Sym("0"), Sym("1")},
        seed: int = 42,
        randomize: bool = True,
    ):
        self.seed = seed

        self.Σ = Σ
        self.Γ = Γ

        self.stack = [BOT]
        self.δ = {}
        if randomize:
            self._random_δ()

    def step(self, sym: Sym):
        assert sym in self.Σ

        γ_top = self.stack[-1]
        action, γ_new = self.δ[sym][γ_top]

        if action == Action.PUSH:
            self.stack.append(γ_new)
        elif action == Action.POP:
            self.stack.pop()
        elif action == Action.NOOP:
            pass
        else:
            raise Exception

    def accept(self, y: String):
        for sym in y:
            self.step(sym)

        return self.stack == [BOT]

    def _random_δ(self):
        random.seed(self.seed)

        pushes = {(Action.PUSH, γ) for γ in self.Γ if γ != BOT}

        for sym in self.Σ:
            if sym not in self.δ:
                self.δ[sym] = {}
            for γ in self.Γ:
                if γ == BOT:
                    flip = random.randint(0, 1)
                    if flip == 0:
                        self.δ[sym][γ] = (Action.NOOP, γ)
                    else:
                        self.δ[sym][γ] = random.choice(list(pushes))
                else:
                    flip = random.randint(0, 2)
                    if flip == 0:
                        self.δ[sym][γ] = (Action.NOOP, γ)
                    elif flip == 1:
                        self.δ[sym][γ] = (Action.POP, γ)
                    else:
                        self.δ[sym][γ] = random.choice(list(pushes))


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
