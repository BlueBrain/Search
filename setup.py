from setuptools import setup, find_packages

VERSION = '0.1'

description = "Blue Brain Search"

install_requires = [
    'Flask',
    'ipywidgets',
    'numpy',
    'pandas',
    'pdfkit',
    'requests',
    'scibert @ git+https://github.com/allenai/scibert',
    'scikit-learn',
    'scipy',
    'sent2vec-prebuilt',
    'sentence-transformers',
    'scispacy',
    'en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz',
    'en-ner-craft-md @ https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz',
    'spacy==2.2.1',
    'tensorflow',
    'tensorflow_hub',
    'torch',
    'transformers',
    'rdflib-jsonld',
    'faiss-cpu',
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
            "create_database=bbsearch.database_entrypoint:main",
            "compute_embeddings=bbsearch.embeddings_entrypoint:main",
        ]
    }
)
