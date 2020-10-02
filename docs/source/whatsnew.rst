Changelog
=========
This page containst changelogs for Blue Brain Search released versions.

Version 0.0.3
-------------
**October 2, 2020**

- This is the first beta release from Blue Brain Search.
- Previous releases were highly experimental and should be considered as being
  in alpha phase.

Changes
^^^^^^^
- **Updgrade** CORD19 database from v35 to v47.
- **Add** button to Literature Search widget to let user choose whether to retrieve
  top N articles or top N sentences.
- **Fix** bug in database creation where auto-increment was triggered even if
  insertion failed.
- **Add** automatic creation of a `FULLTEXT INDEX` on `sentences.text` when
  the table is first created, just after data insertion.
- **Add** annotations for NER with DVC.
- **Add** pipelines to train and evaluate NER models with DVC.
- **Add** `Sent2VecModel` class and option in Literature Search widget to select
  sent2vec to run the search.
- **Improve** Docker ecosystem with `.env` files and `docker-compose`.
- **Refactor** search servers by merging `RemoteSearcher` and `LocalSearcher`
  into the new `SearchEngine`.




