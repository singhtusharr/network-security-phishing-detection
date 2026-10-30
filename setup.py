"""Packages `networksecurity` as an installable module (`pip install -e .`)
so it can be imported from anywhere (scripts, tests, the FastAPI app, Docker)
without sys.path hacks, and so its third-party dependencies are declared in
one place.
"""
from setuptools import find_packages, setup


def get_requirements(file_path: str = "requirements.txt") -> list[str]:
    with open(file_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if "-e ." in requirements:
        requirements.remove("-e .")
    return requirements


setup(
    name="networksecurity",
    version="0.0.1",
    author="Tushar Singh",
    author_email="ts1691281@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)
