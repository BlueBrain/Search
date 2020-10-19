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


Verions 0.0.4
=============
**Ocober 20, 2020**

Changes
-------
- |Add| language detection with `langdetect`, allowing to filter out articles
  not in English or no useful content.
- |Add| widgets inform the user on the CORD-19 version being used.
- |Add| `bbsearch.utils.JSONL` for easy interaction with JSONL files.
- |Add| `bbsearch.entity.PatternCreator` and other functionalities to perform
  rule-based named entity recognition.
- |Change| module names
- |Change| in `bbsearch.embedding_models`, `SBERT` class is now replaced by a
  more general-purpose `SentTransformer` which can wrap any object from
  `sentence_transformers.SentenceTransformer`.
- |Add| `bbsearch.embedding_models.SklearnVectorizer` is a new class that can be used to wrap any `sklearn`
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




