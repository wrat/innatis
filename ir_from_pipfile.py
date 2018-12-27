from pip._internal import main as pip
import os

from install_requires import convert


pip(['install', 'install-requires[pipfile]', '-q'])

path = os.getcwd()
install_requires, dependency_links = convert('Pipfile', 'setup.py', path)
print(install_requires)
