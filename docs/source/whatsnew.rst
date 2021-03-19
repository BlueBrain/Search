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



