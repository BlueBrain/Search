.. Blue Brain Search is a text mining toolbox focused on scientific use cases.
   Copyright (C) 2020  Blue Brain Project, EPFL.
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public License
   along with this program. If not, see <https://www.gnu.org/licenses/>.

.. include:: ./_substitutions.rst

*********
Changelog
*********
This page contains changelogs for Blue Brain Search released versions.

Legend
======
- |Add| denotes new features.
- |Fix| denotes bug fixes.
- |Change| denotes functionality changes.
- |Deprecate| denotes deprecated features that will be removed in the future.
- |Remove| denotes removed features.


Latest
======
- |Change| :code:`uid` generation raises :code:`ValueError` if all identifiers
  are :code:`None`.
- |Add| code to download :code:`arxiv` papers from a given date.
- |Change| the behaviour of the entrypoint :code:`bbs_database download` when the
  specified :code:`--from-month` is too old and the source changed its structure of storing articles
  meanwhile. Now print an error and exit.
- |Add| code to download :code:`PMC` papers from a given date.
- |Add| entrypoint :code:`bbs_database download`.
- |Add| run the tox env ``check-apidoc`` in CI
- |Add| tox environments ``apidoc`` and ``check-apidoc``
- |Add| input type ``tei-xml`` for the ``bbs_database parse`` command.
- |Add| option ``--dry-run`` for ``bbs_database parse`` to display files to
  parse without parsing them.
- |Add| option ``--recursive`` for ``bbs_database parse`` to parse files
  recursively.
- |Add| option ``--match-filename`` for ``bbs_database parse`` to parse only
  files with a name matching a given regular expression.
- |Change| split the CI job into smaller jobs
- |Change| for :code:`bbs_database parse` the value for :code:`input_type` from
  :code:`pmc-xml` to :code:`jats-xml`.
- |Change| name for :code:`PMCXMLParser` to :code:`JATSXMLParser`.
- |Add| article parser for TEI XML files.
- |Add| CLI subcommand ``bbs_database convert-pdf``.
- |Add| parsing of PDFs through a GROBID server.
- |Add| default value :code:`None` for optional fields of :code:`Article`.
- |Add| loading of metadata and abstracts from :code:`PubMed`.
- |Fix| parsing in :code:`PubMed` metadata of authors with a
  :code:`<CollectiveName>` instead of a :code:`<LastName>`.
- |Add| an :code:`ArticleParser` for metadata and abstracts from :code:`PubMed`.
- |Change| the behaviour of :code:`bbs_database add` when no article was loaded
  from the given path. Now, stop with a :code:`RuntimeWarning` and don't load
  the NLP model to get sentences (fail faster).
- |Change| the behaviour of :code:`bbs_database add` when no sentence was
  extracted from the given path, Now, stop with a :code:`RuntimeWarning`.
- |Change| serialization of processed articles from Pickle to JSON format.
- |Add| command line entrypoints :code:`bbs_database init`,
  :code:`bbs_database parse`, and :code:`bbs_database add` to initialize
  a literature database, parse, and integrate articles.
- |Add| research of topic at journal and article levels in :code:`topic` module.
- |Add| :code:`PMCXMLParser` to parse PubMed articles in XML JATS format.
- |Fix| DVC pipeline named :code:`sentence_embedding` regarding missing
  :code:`deps` elements and mixed models origin.
- |Fix| the incorrect maximum input length to the transformer model used as
  backbone for the NER models.
- |Add| :code:`BioBERT NLI+STS CORD-19 v1` building script as a DVC pipeline.
- |Fix| the incorrect maximum input length to the transformer model used as
  backbone for the sentence embedding model :code:`BioBERT NLI+STS CORD-19 v1`.
- |Add| deterministic generation of paper UIDs based on paper identifiers.
- |Change| relative imports into absolute ones.
- |Add| the tables :code:`articles` and :code:`sentences` for
  :code:`bbs_database init` and :code:`bbs_database add`.


Version 0.2.0
==============
**July 1, 2021**

- |Add| metrics file resulting from :code:`dvc` pipelines to :code:`git`.
  This allow now to use :code:`dvc metrics diff`.
- |Change| dependencies required to run the code of :code:`data_and_models/` are not
  installed by default and now require :code:`pip install .[data_and_models]`.
- |Add| in :code:`dvc`, in :code:`ner` pipelines, scripts allowing to train and evaluate
  NER thanks to the :code:`huggingface/transformers` package.
  A comparison with :code:`spaCy` training is also possible.
- |Change| reports format of Search Widget from PDF to HTML.
- |Remove| :code:`tqdm`, :code:`joblib`, :code:`pdfkit` dependencies.
- |Remove| :code:`bluesearch.mining.eval.plot_ner_confusion_matrix` function to drop
  :code:`joblib` from :code:`install_requires`.
- |Change| :code:`requirements.txt` refactored into three separate lists of
  dependencies: :code:`requirements.txt`, :code:`requirements-dev.txt`,
  :code:`requirements-data_and_models.txt`.
- |Fix| bugs (related to nested entities) in :code:`ner_report`, :code:`ner_errors`,
  :code:`ner_confusion_matrix` functions from :code:`bluesearch.mining.eval` submodule.
- |Add| utility function :code:`_check_consistent_iob` inside
  :code:`bluesearch.mining.eval`.
- |Change| upgrade linting tools in ``tox.ini``
- |Change| for Transformer-based :code:`spaCy` pipelines for NER models
  instead of Tok2Vec-based :code:`scispaCy` pipelines.
- |Change| for one entity per model instead of several entities per NER model.
- |Change| :code:`pipelines/ner/dvc.yaml` to simplify and harmonize the
  definition of the pipeline for training NER models.
- |Add| :code:`annotations/ner/analyze.py`, a code to evaluate the data quality
  of annotations. It could generate: 1) a detailed report for individual files
  when used as a script and 2) a summary table for several files when used as
  a function.
- |Add| :code:`pipelines/ner/clean.py`, a script to clean annotations. It keeps
  only valid texts, normalizes labels, keeps only a given label, and then
  renames the label if necessary.
- |Remove| :code:`ee_models_library.csv` and change the logic for one model per entity type.
- |Add| :code:`ArticleParser` abstract class representing a generic interface
  for parsing articles.
- |Add| :code:`CORD19ArticleParser` to parse CORD-19 articles in JSON format.

Version 0.1.2
=============
- |Change| spaCy version from 2.x to 3.x, including scispaCy and models versions.
- |Change| the training of NER models: use spaCy directly instead of Prodigy,
  use the default configuration from spaCy 3 instead of from Prodigy, use the
  binary format (:code:`.spacy`) from spaCy 3 instead of the :code:`.jsonl`
  format from Prodigy.
- |Remove| Prodigy dependency.


Version 0.1.1
=============
- |Change| Upgrade to :code:`dvc 2.0`.
- |Remove| NLTK dependencies.
- |Change| Drop the dedicated :code:`SBioBERT` class, we now use
  :code:`SentTransformer` interface to support this model.


Version 0.1.0
=============
- |Add| in :code:`dvc` pipelines, the :code:`Dockerfile` now installs
  `requirements.txt` to fix the versions of dependencies. 
- |Add| support for :code:`Python 3.9`.
- |Add| Blue Brain Search as a Zenodo record. This provides a unique DOI, a DOI
  for each published release, and automatic preservation outside GitHub.
- |Add| the content of the DVC remote for Blue Brain Search v0.1.0 as a Zenodo
  record. This provides DOIs as for the code of Blue Brain Search above. This
  is also the first public release of the data and models of Blue Brain Search.
- |Remove| support for :code:`Python 3.6`.
- |Remove| the external dependency :code:`sent2vec` and the embedding models
  depending on it, i.e. :code:`BSV` and :code:`Sent2VecModel`.
- |Remove| the embedding model :code:`Universal Sentence Encoder`: (USE) and its
  dependencies (:code:`tensorflow` and :code:`tensorflow-hub`).
- |Remove| :code:`BBS_BBG_poc` notebook (now hosted on
  https://github.com/BlueBrain/Search-Graph-Examples) and :code:`assets/`
  directory.


Version 0.0.10
==============

Changes
-------
- |Change| :code:`bluesearch` is the new name of the Python package, replacing the former :code:`bbsearch`.
- |Change| The code is now hosted on GitHub under :code:`BlueBrain/Search`, eliminating the redundancy of the former :code:`BlueBrain/BlueBrainSearch`.
- |Add| in `README` the purpose of Blue Brain Search.
- |Add| in `README` the common usage of the two widgets (search and mining).
- |Add| in `README` a complete and step-by-step *Getting Started*.
- |Add| type checking for third-party libraries (:code:`NumPy`, :code:`Pandas`, :code:`SQLAlchemy`).
- |Add| :code:`BioBERT NLI+STS CORD-19 v1` to DVC evaluation pipeline.


Version 0.0.9
=============
**December 11, 2020**

Changes
-------
- |Add| saving and loading of the results from the literature search and mining widgets.
- |Add| mining for more than 1,000 articles.
- |Add| :code:`BioBERT NLI+STS CORD-19 v1` training scripts and data.
- |Add| CORD-19 version 65 database, embeddings, and entities.
- |Add| tests for all entry points.
- |Add| security checks with :code:`bandit`.
- |Fix| NER false positive for :code:`abstract`.
- |Fix| refactoring issue in :code:`get_embedding_model`.
- |Change| naming of and inside the :code:`bluesearch.entrypoints` module.
- |Change| how the NER entry points retrieve models: now DVC is used.
- |Change| warnings when generating the documentation into errors.
- |Remove| :code:`scibert` from :code:`setup.py` and :code:`requirements.txt`.


Version 0.0.8
=============
**November 24, 2020**

Changes
-------
- |Add| column `is_bad` in table `sentences` for quality filtering (too long, too short, LaTeX code).
- |Add| embedding model `BioBERT NLI+STS CORD-19 v1`.
- |Change| `embedding_models.get_embedding_model()` to support any model class and checkpoint path without having to
  modify the source code of BBS.
- |Fix| bug in hyperlinks of SearchWidget. We now take the first URL if there are several, and add Google search if
  there is none.
- |Change| widgets UIs with tabs to improve usability.


Version 0.0.7
=============
**November 16, 2020**

Changes
-------
- |Add| parallelization of embedding computations.
- |Change| "Saved Articles" summary in the Search Widget.
- |Fix| undesired timeout of MySQL connection in the Search Server.


Version 0.0.6
=============
**November 3, 2020**

Changes
-------
- |Add| inter-rater agreement with DVC.
- |Add| Advanced Features section in the Search Widget.
- |Change| mining schema logic.
- |Change| code formatting - run `black` on everything.


Version 0.0.5
=============
**October 26, 2020**

Changes
-------
- |Change| `bluesearch.mining.eval.spacy2df` can now work with NER pipelines
  including entity rulers.


Version 0.0.4
=============
**October 20, 2020**

Changes
-------
- |Add| language detection with `langdetect`, allowing to filter out articles
  not in English or no useful content.
- |Add| widgets inform the user on the CORD-19 version being used.
- |Add| `bluesearch.utils.JSONL` for easy interaction with JSONL files.
- |Add| `bluesearch.entity.PatternCreator` and other functionalities to perform
  rule-based named entity recognition.
- |Change| module names
- |Change| in `bluesearch.embedding_models`, `SBERT` class is now replaced by a
  more general-purpose `SentTransformer` which can wrap any object from
  `sentence_transformers.SentenceTransformer`.
- |Add| `bluesearch.embedding_models.SklearnVectorizer` is a new class that can be used to wrap any `sklearn`
  vectorizer object (`TfidfVectorizer`, `CountVectorizer`, `HashingVectorizer`).


Version 0.0.3
=============
**October 2, 2020**

- This is the first beta release from Blue Brain Search.
- Previous releases were highly experimental and should be considered as being
  in alpha phase.

Changes
-------
- |Change| CORD19 database version, upgrading from v35 to v47.
- |Add| button to Literature Search widget to let user choose whether to retrieve
  top N articles or top N sentences.
- |Fix| bug in database creation where auto-increment was triggered even if
  insertion failed.
- |Add| automatic creation of a `FULLTEXT INDEX` on `sentences.text` when
  the table is first created, just after data insertion.
- |Add| annotations for NER with DVC.
- |Add| pipelines to train and evaluate NER models with DVC.
- |Add| `Sent2VecModel` class and option in Literature Search widget to select
  sent2vec to run the search.
- |Add| Docker ecosystem with `.env` files and `docker-compose`.
- |Change| search servers by merging `RemoteSearcher` and `LocalSearcher`
  into the new `SearchEngine`.


