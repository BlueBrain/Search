# BBSearch is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

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
    "langdetect",
    "matplotlib",
    "mysqlclient",
    "networkx",
    "nexusforge @ git+https://github.com/BlueBrain/nexus-forge.git",
    "nltk",
    "numpy>=1.16.1",
    "pandas>=1.0.0",
    "pdfkit",
    "pymysql",
    "python-dotenv",
    "rdflib-jsonld",
    "requests",
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
    "pyarrow",
]

extras_require = {
    "dev": [
        "cryptography",
        "docker",
        "flake8",
        "mypy",
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
    author="Blue Brain Project, EPFL",
    url="https://github.com/BlueBrain/BlueBrainSearch",
    project_urls={
        "Source": "https://github.com/BlueBrain/BlueBrainSearch",
        "Documentation": "https://bbpteam.epfl.ch/documentation",
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/BBS",
    },
    license="-",
    use_scm_version={
        "write_to": "src/bbsearch/version.py",
        "write_to_template": '"""The package version."""\n__version__ = "{version}"\n',
        # "local_scheme": "no-local-version",
    },
    package_dir={"": "src"},
    packages=find_packages("./src"),
    package_data={"bbsearch": ["_css/stylesheet.css", "py.typed"]},
    zip_safe=False,
    python_requires=">=3.6",
    setup_requires=["setuptools_scm"],
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "compute_embeddings = bbsearch.entrypoint:run_compute_embeddings",
            "create_database = bbsearch.entrypoint:run_create_database",
            "create_mining_cache = bbsearch.entrypoint:run_create_mining_cache",
            "embedding_server = bbsearch.entrypoint:run_embedding_server",
            "mining_server = bbsearch.entrypoint:run_mining_server",
            "search_server = bbsearch.entrypoint:run_search_server",
        ]
    },
)
