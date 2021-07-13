from setuptools import find_packages, setup
setup(
    name='DenStreamLib',
    packages=find_packages(include=['microCluster']),
    version='0.1.0',
    description='DenStream library',
    author='yuriPedro',
    license='MIT',
    install_requires=['numpy','sklearn'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1','pandas'],
    test_suite='tests',
)
