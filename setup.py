from setuptools import setup, find_packages

setup(
      name='poissonpy',
      version='1.0.1',
      description='Plug-and-play 2D Poisson equations library.',
      long_description="Plug-and-play 2D Poisson equation library. Useful in scientific computing, image and video processing, computer graphics.",
      url='https://github.com/astro-jingtao/poissonpy',
      keywords = "pde poisson-equation laplace-equation scientific-computing poisson",
      author='bchao1, Tao Jing',
      license='MIT',
      packages=find_packages(),
      python_requires=">=3.6"
)