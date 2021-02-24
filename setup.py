# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from setuptools import find_packages, setup

DESCRIPTION = (
    "Blue Brain text mining toolbox for semantic search and information extraction"
)

LONG_DESCRIPTION = """
Blue Brain Search is a text mining toolbox to perform semantic literature search 
and structured information extraction from text sources.

This project originated from the Blue Brain Project efforts on exploring and 
mining the CORD-19 dataset."""

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    "Operating System :: Unix",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Text Processing :: General",
]

PYTHON_REQUIRES = ">=3.7, <=3.9"

INSTALL_REQUIRES = [
    "Flask",
    "SQLAlchemy",
    "dvc[ssh]",
    "h5py",
    "ipython",
    "ipywidgets",
    "langdetect",
    "matplotlib",
    "mysqlclient",
    "networkx",
    "nltk",
    "numpy>=1.20.1",
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
    "torch",
    "tqdm",
    "transformers==3.0.2",
    "pyarrow",
]

EXTRAS_REQUIRE = {
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
    ],
}

setup(
    name="bluesearch",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Blue Brain Project, EPFL",
    url="https://github.com/BlueBrain/Search",
    project_urls={
        "Source": "https://github.com/BlueBrain/Search",
        "Documentation": "https://bbpteam.epfl.ch/documentation",
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/BBS",
    },
    license="-",
    classifiers=CLASSIFIERS,
    use_scm_version={
        "write_to": "src/bluesearch/version.py",
        "write_to_template": '"""The package version."""\n__version__ = "{version}"\n',
        "local_scheme": "no-local-version",
    },
    package_dir={"": "src"},
    packages=find_packages("./src"),
    package_data={"bluesearch": ["_css/stylesheet.css", "py.typed"]},
    zip_safe=False,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "compute_embeddings = bluesearch.entrypoint:run_compute_embeddings",
            "create_database = bluesearch.entrypoint:run_create_database",
            "create_mining_cache = bluesearch.entrypoint:run_create_mining_cache",
            "embedding_server = bluesearch.entrypoint:run_embedding_server",
            "mining_server = bluesearch.entrypoint:run_mining_server",
            "search_server = bluesearch.entrypoint:run_search_server",
        ]
    },
)
