from setuptools import find_packages, setup


install_requires = [
    'Flask',
    'ipywidgets',
    'h5py',
    'matplotlib',
    'numpy>=1.16.1',
    'pandas',
    'pdfkit',
    'requests',
    'scibert @ git+https://github.com/allenai/scibert',
    'scikit-learn',
    'scipy',
    'sent2vec-prebuilt',
    'sentence-transformers',
    'scispacy',
    'en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz',
    'en-ner-craft-md @ https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_craft_md-0.2.5.tar.gz',
    'spacy==2.3.1',
    'SQLAlchemy',
    'tensorflow',
    'tensorflow_hub',
    'torch',
    'tqdm',
    'transformers',
    'rdflib-jsonld',
    'faiss-cpu',
    'jupyter_server_proxy',
    'jupyter_dash',
    'networkx',
    'dash-cytoscape',
    'dash-table',
    'nexusforge @ git+https://github.com/BlueBrain/nexus-forge.git'
]
tests_require = [
    'flake8',
    'pytest',
    'pytest-cov',
]

extras_require = {'dev': ['flake8', 'pydocstyle', 'pytest>=4.6', 'pytest-cov']}

setup(
    name='BBSearch',
    description='Blue Brain Search',
    author='Blue Brain Project',
    use_scm_version={
        'write_to': 'src/bbsearch/_version.py',
        'write_to_template': '__version__ = "{version}"\n',
        # 'local_scheme': "no-local-version",
    },
    package_dir={'': 'src'},
    packages=find_packages("./src"),
    python_requires='>=3.6',
    setup_requires=['setuptools_scm'],
    install_requires=install_requires,
    extras_require=extras_require,
    tests_require=tests_require,
    entry_points={
        "console_scripts": [
            "embedding_server=bbsearch.entrypoints.embedding_server_entrypoint:main",
            "create_database=bbsearch.entrypoints.database_entrypoint:main",
            "compute_embeddings=bbsearch.entrypoints.embeddings_entrypoint:main",
            "search_server=bbsearch.entrypoints.search_server_entrypoint:main",
            "mining_server=bbsearch.entrypoints.mining_server_entrypoint:main",
        ]
    }
)
