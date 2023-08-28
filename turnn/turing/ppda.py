from enum import IntEnum, unique
from itertools import product
from math import log
from typing import List, Tuple, Union

import numpy as np

from turnn.base.string import String
from turnn.base.symbol import BOT, Sym


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

        # δ: Σ × Γ → (({PUSH, POP, NOOP} × Γ) × R)
        self.δ = {sym: {γ: {} for γ in self.Γ} for sym in self.Σ}
        if randomize:
            self._random_δ()

    def step(self, stack: List[Sym], a: Sym) -> Tuple[List[Sym], float]:
        assert a in self.Σ

        γ = stack[-1]
        action, γʼ = self.δ[a][γ][0]

        # Modify the stack according to the action.
        if action == Action.PUSH:
            stack.append(γʼ)
        elif action == Action.POP:
            stack.pop()
        elif action == Action.NOOP:
            pass
        else:
            raise Exception

        # Return the new stack and the probability of the action.
        return stack, self.δ[a][γ][1]

    @property
    def probabilistic(self):
        """Checks if the PDA is probabilistic."""
        d = {γ: 0 for γ in self.Γ}
        for a, γ in product(self.Σ, self.Γ):
            d[γ] += self.δ[a][γ][1]
        return all(abs(d[γ] - 1) < 1e-6 for γ in self.Γ)

    def accept(self, y: Union[str, String]) -> Tuple[bool, float]:
        """Computes the acceptance probability of a string. Returns a tuple of
        the acceptance status and the log probability.

        Args:
            y (Union[str, String]): The string to be accepted.

        Returns:
            Tuple[bool, float]: The acceptance status and the log probability.
        """
        if isinstance(y, str):
            y = String(y)

        stack = [BOT]
        logp = 0

        # Simulate a run of the PDA. (Assumes that the PDA is deterministic.)
        for a in y:
            stack, p = self.step(stack, a)
            logp += log(p)

        return stack == [BOT], logp

    def __call__(self, y: Union[str, String]) -> Tuple[bool, float]:
        """Computes the acceptance probability of a string. Returns a tuple of
        the acceptance status and the log probability.
        It simply calls the accept method.

        Args:
            y (Union[str, String]): The string to be accepted.

        Returns:
            Tuple[bool, float]: The acceptance status and the log probability.
        """
        return self.accept(y)

    def _random_δ(self):
        """Initializes a random transition function and with it a random PPDA."""
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
        randomize: bool = True,
    ):
        self.seed = seed

        self.Σ = Σ
        self.Γ_1 = Γ_1
        self.Γ_2 = Γ_2

        # δ: Σ × (Γ_1 × Γ_2) → (({PUSH, POP, NOOP} × Γ_1 × Γ_2) × R)
        self.δ = {
            sym: {(γ_1, γ_2): {} for (γ_1, γ_2) in product(self.Γ_1, self.Γ_2)}
            for sym in self.Σ
        }
        if randomize:
            self._random_δ()

    def _execute_action(self, stack: List[Sym], action: Action, γ_new: Sym):
        """Commits an action to a the current stack configuration."""
        if action == Action.PUSH:
            stack.append(γ_new)
        elif action == Action.POP:
            stack.pop()
        elif action == Action.NOOP:
            pass
        else:
            raise Exception

    def step(
        self, stacks: Tuple[List[Sym], List[Sym]], a: Sym
    ) -> Tuple[Tuple[List[Sym], List[Sym]], float]:
        """Executes a step of the PDA. Returns a tuple of the new stacks and the
        probability of the action.

        Args:
            stacks (Tuple[List[Sym], List[Sym]]): The current stacks.
            a (Sym): The current symbol.

        Returns:
            Tuple[Tuple[List[Sym], List[Sym]], float]: The new stacks and the
                probability of the action.
        """
        assert a in self.Σ

        γ_1, γ_2 = stacks[0][-1], stacks[1][-1]
        (action_1, γ_1ʼ), (action_2, γ_2ʼ) = self.δ[a][(γ_1, γ_2)][:2]

        self._execute_action(stacks[0], action_1, γ_1ʼ)
        self._execute_action(stacks[1], action_2, γ_2ʼ)

        return stacks, self.δ[a][(γ_1, γ_2)][2]

    def accept(self, y: Union[str, String]) -> Tuple[bool, float]:
        """Computes the acceptance probability of a string. Returns a tuple of
        the acceptance status and the log probability.

        Args:
            y (Union[str, String]): The string to be accepted.

        Returns:
            Tuple[bool, float]: The acceptance status and the log probability.
        """
        if isinstance(y, str):
            y = String(y)

        stacks = [BOT], [BOT]
        logp = 0

        for a in y:
            stacks, p = self.step(stacks, a)
            logp += log(p)

        return stacks[0] == [BOT] and stacks[1] == [BOT], logp

    def __call__(self, y: Union[str, String]) -> Tuple[bool, float]:
        """Computes the acceptance probability of a string. Returns a tuple of
        the acceptance status and the log probability.
        It simply calls the accept method.

        Args:
            y (Union[str, String]): The string to be accepted.

        Returns:
            Tuple[bool, float]: The acceptance status and the log probability.
        """
        return self.accept(y)

    def _get_action(
        self, γ_top: Sym, pushes: List[Tuple[Action, Sym]], rng: np.random.Generator
    ):
        """Returns a random action and a random symbol for a given stack top."""
        if γ_top == BOT:
            flip = rng.integers(0, 1, endpoint=True)
            if flip == 0:
                action, γ_new = (Action.NOOP, γ_top)
            else:
                action, γ_new = tuple(rng.choice(pushes))
        else:
            flip = rng.integers(0, 2, endpoint=True)
            if flip == 0:
                action, γ_new = (Action.NOOP, γ_top)
            elif flip == 1:
                action, γ_new = (Action.POP, γ_top)
            else:
                action, γ_new = tuple(rng.choice(pushes))

        return action, γ_new

    def _random_δ(self):
        """Initializes a random transition function and with it a random PPDA."""
        rng = np.random.default_rng(self.seed)

        pushes = [
            [(Action.PUSH, γ) for γ in Γ] for Γ in [self.Γ_1 - {BOT}, self.Γ_2 - {BOT}]
        ]

        for γ_1, γ_2 in product(self.Γ_1, self.Γ_2):
            α = rng.dirichlet(np.ones(len(self.Σ)))
            for ii, a in enumerate(self.Σ):
                action_1, γ_1ʼ = self._get_action(γ_1, pushes[0], rng)
                action_2, γ_2ʼ = self._get_action(γ_2, pushes[1], rng)

                self.δ[a][(γ_1, γ_2)] = (action_1, γ_1ʼ), (action_2, γ_2ʼ), α[ii]
