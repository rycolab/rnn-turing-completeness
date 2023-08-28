from typing import List

from turnn.base.symbol import BOT, Sym


def cantor_decode(x) -> List[Sym]:
    """
    This simply decodes the value of the cell in the hidden state representing a stack
    into a sequence of symbols.

    Args:
        x: The value to be decoded into a sequence of symbols.

    Returns:
        List[Sym]: The decoded sequence of symbols.
    """
    stack = [BOT]
    if x.p == 0:
        return stack
    return stack + list(
        map(lambda x: Sym("0") if x == "1" else Sym("1"), reversed(list(str(x.p))))
    )
