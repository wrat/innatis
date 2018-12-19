from distutils.core import setup
setup(
  name='innatis',
  packages=['innatis'], # this must be the same as the name above
  version='0.1',
  description='A library of useful custom Rasa components',
  long_description=open('README.md').read(),
  author='CarLabs',
  author_email='blake@carlabs.com',
  url='https://github.com/Revmaker/innatis',
  download_url='https://github.com/Revmaker/innatis/tarball/0.1',
  keywords=['rasa', 'nlu', 'components'],
  classifiers=[]
)
