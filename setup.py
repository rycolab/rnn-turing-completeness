from setuptools import setup

install_requires = [
    "numpy",
    "sympy",
    # needed for the tests
    "pytest",
]


setup(
    name="turnn",
    install_requires=install_requires,
    version="0.1",
    scripts=[],
    packages=["turnn"],
)
