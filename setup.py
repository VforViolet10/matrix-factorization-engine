from setuptools import setup, find_packages


setup(
    name="matrix-factorization-engine",
    version="0.1.0",
    description=(
        "A research-oriented engine for matrix "
        "factorization and decomposition."
    ),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.3",
    ],
)
