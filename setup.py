import os

from setuptools import setup

path = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(path, "requirements.txt"), encoding="utf-8") as f:
        REQUIRED = f.read().split("\n")
except FileNotFoundError:
    REQUIRED = []

setup(
    name="ner_playground",
    version="0.0.1",
    author="Youness MANSAR",
    author_email="mansaryounessecp@gmail.com",
    description="nlp",
    license="GNU",
    keywords="nlp",
    url="https://github.com/CVxTz/XXXX",
    package_dir={"": "src"},
    packages=["ner_playground"],
    data_files=[("ner_playground", ["*.json"])],
    classifiers=[
        "Topic :: Utilities",
    ],
    install_requires=REQUIRED,
)
