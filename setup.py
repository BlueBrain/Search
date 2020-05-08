from setuptools import setup, find_packages

VERSION = '0.1'

description = "Blue Brain Search"

install_requires = [
    'absl-py',
    'Flask',
    'nltk',
    'numpy',
    'pandas',
    'requests',
    'scikit-learn',
    'scipy',
    'sent2vec-prebuilt',
    'sentence-transformers',
    'spacy',
    'tensorflow>=2.0.0a0',
    'tensorflow_hub',
    'torch'
]

setup_requires = ['pytest-runner']
tests_require = [
    'flake8',
    'pytest',
    'pytest-cov',
]

extras_require = {'dev': ['flake8', 'pydocstyle', 'pytest', 'pytest-cov']}

setup(
    name="BBSearch",
    description=description,
    author='Blue Brain Project',
    version=VERSION,
    package_dir={'': 'src'},
    packages=find_packages("./src"),
    python_requires='>=3.6',
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    tests_require=tests_require,
    entry_points={
        "console_scripts": [
            "embedding_server=bbsearch.server.embedding_server_entrypoint:main",
            "create_database=bbsearch.create_database_entrypoint:main",
            "save_embeddings=bbsearch.save_embeddings_entrypoint:main"
        ]
    }
)
