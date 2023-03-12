from typing import Union, List, Set

from turnn.base.symbol import Sym, ε, to_sym


class Alphabet:
    def __init__(self, symbols: Union[List[Sym], Set[Sym]]):
        assert len(symbols) > 0, "Alphabet must be non-empty."
        assert ε not in symbols, "ε must not be in the alphabet."
        self.symbols = symbols if isinstance(symbols, set) else set(symbols)

    def __str__(self):
        return "".join(str(sym) for sym in self.symbols)

    def __repr__(self):
        return "".join(str(sym) for sym in self.symbols)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, Alphabet) and self.symbols == other.symbols

    def __len__(self):
        return len(self.symbols)

    def __contains__(self, item):
        return item in self.symbols

    def __iter__(self):
        return iter(self.symbols)


def to_alphabet(
    symbols: Union[List[Union[Sym, str]], Set[Union[Sym, str]]]
) -> Alphabet:
    """Creates an Alphabet from a list of symbols.

    Args:
        symbols (Union[List[Union[Sym, str]], Set[Union[Sym, str]]]): _description_

    Returns:
        Alphabet: _description_
    """

    if isinstance(symbols, Alphabet):
        return symbols
    else:
        return Alphabet(
            {to_sym(sym) if isinstance(sym, str) else sym for sym in symbols}
        )
