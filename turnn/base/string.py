from typing import List, Union

from turnn.base.symbol import Sym, to_sym


class String:
    def __init__(self, y: Union[str, List[Sym]]):
        self.y = y if isinstance(y, list) else [to_sym(sym) for sym in y]

    def __str__(self):
        return "".join(str(sym) for sym in self.y)

    def __repr__(self):
        return "".join(str(sym) for sym in self.y)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, String) and self.y == other.y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, key):
        return self.y[key]

    def __iter__(self):
        return iter(self.y)
