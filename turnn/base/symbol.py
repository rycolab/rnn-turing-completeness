class Sym:
    def __init__(self, sym):
        self.sym = sym

    def __str__(self):
        return str(self.sym)

    def __repr__(self):
        return str(self.sym)

    def __hash__(self):
        return hash(self.sym)

    def __eq__(self, other):
        return isinstance(other, Sym) and self.sym == other.sym

    def __invert__(self):
        return self


ε = Sym("ε")

# String sybols
BOS = Sym("BOS")
EOS = Sym("EOS")

# Stack symbols
BOT = Sym("⊥")


def to_sym(s: str) -> Sym:
    """Converts a single character string to a symbol (Sym).

    Args:
        s (str): The input string

    Returns:
        Sym: Sym-ed version of the input string.
    """
    if isinstance(s, Sym):
        return s
    else:
        return Sym(s)
