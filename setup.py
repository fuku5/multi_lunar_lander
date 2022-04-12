from os.path import splitext

from setuptools import setup
from setuptools import find_packages


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name='multiple_lunar_lander',
    version="0.0.6",
    description='openai gym lunar lander with changing goals',
    packages=['multi_lunar'],
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file('requirements.txt'),
)
