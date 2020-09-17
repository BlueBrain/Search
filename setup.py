from setuptools import find_packages, setup

install_requires = [
    "Flask",
    "SQLAlchemy",
    "dash-cytoscape",
    "dash-table",
    "dash_daq",
    "dvc[ssh]",
    "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz",
    "en-core-sci-lg @ https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_lg-0.2.5.tar.gz",
    "en-ner-craft-md @ https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_craft_md-0.2.5.tar.gz",
    "faiss-cpu",
    "h5py",
    "ipython",
    "ipywidgets",
    "jupyter_dash",
    "jupyter_server_proxy",
    "matplotlib",
    "mysqlclient",
    "networkx",
    "nexusforge @ git+https://github.com/BlueBrain/nexus-forge.git",
    "numpy>=1.16.1",
    "pandas>=1.0.0",
    "pdfkit",
    "pymysql",
    "rdflib-jsonld",
    "requests",
    "scibert @ git+https://github.com/allenai/scibert",
    "scikit-learn",
    "scipy",
    "scispacy",
    "sent2vec-prebuilt",
    "sentence-transformers==0.3.5",
    "spacy==2.3.1",
    "tensorflow",
    "tensorflow_hub",
    "torch",
    "tqdm",
    "transformers==3.0.2",
]

extras_require = {
    "dev": [
        "cryptography",
        "docker",
        "flake8",
        "pyaml"
        "pydocstyle",
        "pytest>=4.6",
        "pytest-benchmark",
        "pytest-cov",
        "responses",
        "sphinx",
        "sphinx-bluebrain-theme",
        "tox",
    ]
}

setup(
    name="BBSearch",
    description="Blue Brain Search",
    author="Blue Brain Project (EPFL) - ML Team",
    author_email="bbp-ou-machinelearning@groupes.epfl.ch",
    url="https://github.com/BlueBrain/BlueBrainSearch",
    use_scm_version={
        "write_to": "src/bbsearch/version.py",
        "write_to_template": '"""The package version."""\n__version__ = "{version}"\n',
        # "local_scheme": "no-local-version",
    },
    package_dir={"": "src"},
    packages=find_packages("./src"),
    package_data={"bbsearch": ["_css/stylesheet.css"]},
    python_requires=">=3.6",
    setup_requires=["setuptools_scm"],
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "compute_embeddings = bbsearch.entrypoints.embeddings_entrypoint:main",
            "create_database = bbsearch.entrypoints.database_entrypoint:main",
            "create_mining_cache = bbsearch.entrypoints:run_create_mining_cache",
            "embedding_server = bbsearch.entrypoints.embedding_server_entrypoint:main",
            "mining_server = bbsearch.entrypoints.mining_server_entrypoint:main",
            "search_server = bbsearch.entrypoints.search_server_entrypoint:main",
        ]
    },
)
