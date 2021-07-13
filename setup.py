from setuptools import find_packages, setup
setup(
    name='DenStreamLib',
    packages=find_packages(),
    version='0.1.0',
    description='DenStream library',
    author='Yuri Pedro dos Santos',
    author_email='yurisantosypds@gmail.com',
    install_requires=['numpy','sklearn'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
