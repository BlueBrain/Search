"""SQL Related functions."""

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

import logging
from typing import cast

import numpy as np
import pandas as pd
import sqlalchemy.sql as sql


def get_titles(article_ids, engine):
    """Get article titles from the SQL database.

    Parameters
    ----------
    article_ids : iterable of int
        An iterable of article IDs.
    engine : sqlalchemy.engine.Engine
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    titles : dict
        Dictionary mapping article IDs to the article titles.
    """
    if len(article_ids) == 0:
        return {}

    query = sql.text(
        """SELECT article_id, title
        FROM articles
        WHERE article_id IN :article_ids
        """
    )
    query = query.bindparams(sql.bindparam("article_ids", expanding=True))

    with engine.begin() as connection:
        response = connection.execute(query, {"article_ids": article_ids}).fetchall()
        titles = {article_id: title for article_id, title in response}

    return titles


def retrieve_article_ids(engine):
    """Retrieve all articles_id from sentences table.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    article_id_dict : dict
        Dictionary giving the corresponding article_id for a given sentence_id
    """
    result_proxy = engine.execute("SELECT sentence_id, article_id FROM sentences")
    article_id_dict = dict(result_proxy.fetchall())
    return article_id_dict


def retrieve_sentences_from_sentence_ids(sentence_ids, engine, keep_order=False):
    """Retrieve sentences given sentence ids.

    Parameters
    ----------
    sentence_ids : iterable of int
        Sentence ids for which need to retrieve the text.
    engine : sqlalchemy.engine.Engine
        SQLAlchemy Engine connected to the database.
    keep_order : bool, optional
        Make sure that the order of sentence ID in the result data frame
        is the same. Note that the default value is `False`.

    Returns
    -------
    df_sentences : pd.DataFrame
        Pandas DataFrame containing all sentences and their corresponding metadata:
        article_id, sentence_id, section_name, text, paragraph_pos_in_article.
    """
    sql_query = sql.text(
        """
        SELECT article_id, sentence_id, section_name, text, paragraph_pos_in_article
        FROM sentences
        WHERE sentence_id IN :sentence_ids
        """
    )
    sql_query = sql_query.bindparams(sql.bindparam("sentence_ids", expanding=True))

    with engine.begin() as connection:
        df_sentences = pd.read_sql(
            sql_query,
            params={"sentence_ids": [int(id_) for id_ in sentence_ids]},
            con=connection,
        )

    if keep_order:
        # Remove sentence IDs that were not found, otherwise df.loc will fail.
        found_sentence_ids = set(df_sentences["sentence_id"])
        sentence_ids = filter(lambda x: x in found_sentence_ids, sentence_ids)

        # Sort the dataframe by sentence_ids
        df_sentences = (
            df_sentences.set_index("sentence_id").loc[sentence_ids].reset_index()
        )

    return df_sentences


def retrieve_paragraph_from_sentence_id(sentence_id, engine):
    """Retrieve paragraph given one sentence id.

    Parameters
    ----------
    sentence_id : int
        Sentence id for which need to retrieve the paragraph.
    engine : sqlalchemy.engine.Engine
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    paragraph : str or None
        If ``str`` then a paragraph containing the sentence of the given
        sentence_id. If None then the `sentence_id` was not found in the
        sentences table.
    """
    sql_query = sql.text(
        """SELECT text
                    FROM sentences
                    WHERE article_id =
                        (SELECT article_id
                        FROM sentences
                        WHERE sentence_id = :sentence_id )
                    AND paragraph_pos_in_article =
                        (SELECT paragraph_pos_in_article
                        FROM sentences
                        WHERE sentence_id = :sentence_id )
                    ORDER BY sentence_pos_in_paragraph ASC"""
    )

    all_sentences = pd.read_sql(
        sql_query, engine, params={"sentence_id": int(sentence_id)}
    )["text"].to_list()
    if not all_sentences:
        paragraph = None
    else:
        paragraph = " ".join(all_sentences)
    return paragraph


def retrieve_paragraph(article_id, paragraph_pos_in_article, engine):
    """Retrieve paragraph given one identifier (article_id, paragraph_pos_in_article).

    Parameters
    ----------
    article_id : int
        Article id.
    paragraph_pos_in_article : int
        Relative position of a paragraph in an article. Note that the numbering
        starts from 0.
    engine : sqlalchemy.engine.Engine
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    paragraph : pd.DataFrame
        pd.DataFrame with the paragraph and its metadata:
        article_id, text, section_name, paragraph_pos_in_article.
    """
    sql_query = sql.text(
        """SELECT section_name, text
                    FROM sentences
                    WHERE article_id = :article_id
                    AND paragraph_pos_in_article = :paragraph_pos_in_article
                    ORDER BY sentence_pos_in_paragraph ASC"""
    )

    sentences = pd.read_sql(
        sql_query,
        engine,
        params={
            "article_id": int(article_id),
            "paragraph_pos_in_article": int(paragraph_pos_in_article),
        },
    )
    if sentences.empty:
        paragraph = pd.DataFrame(
            columns=["article_id", "text", "section_name", "paragraph_pos_in_article"]
        )
    else:
        sentences_text = sentences["text"].to_list()
        section_name = sentences["section_name"].iloc[0]
        paragraph_text = " ".join(sentences_text)

        paragraph = pd.DataFrame(
            [
                {
                    "article_id": article_id,
                    "text": paragraph_text,
                    "section_name": section_name,
                    "paragraph_pos_in_article": paragraph_pos_in_article,
                },
            ]
        )
    return paragraph


def retrieve_article_metadata_from_article_id(article_id, engine):
    """Retrieve article metadata given one article id.

    Parameters
    ----------
    article_id : int
        Article id for which need to retrieve the article metadata.
    engine : sqlalchemy.engine.Engine
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    article : pd.DataFrame
        DataFrame containing the article metadata. The columns are
        'article_id', 'cord_uid', 'sha', 'source_x', 'title', 'doi',
        'pmcid', 'pubmed_id', 'license', 'abstract', 'publish_time',
        'authors', 'journal', 'mag_id', 'who_covidence_id', 'arxiv_id',
        'pdf_json_files', 'pmc_json_files', 'url', 's2_id'.
    """
    sql_query = sql.text(
        """SELECT *
                    FROM articles
                    WHERE article_id = :article_id"""
    )
    article = pd.read_sql(sql_query, engine, params={"article_id": int(article_id)})
    return article


def retrieve_articles(article_ids, engine):
    """Retrieve article given multiple article ids.

    Parameters
    ----------
    article_ids : list of int
        List of Article id for which need to retrieve the entire text article.
    engine : sqlalchemy.engine.Engine
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    articles : pd.DataFrame
        DataFrame containing the articles divided into paragraphs. The columns are
        'article_id', 'paragraph_pos_in_article', 'text', 'section_name'.
    """
    article_ids = [int(id_) for id_ in article_ids]
    sql_query = sql.text(
        """SELECT *
                    FROM sentences
                    WHERE article_id IN :articles_ids
                    ORDER BY article_id ASC,
                    paragraph_pos_in_article ASC,
                    sentence_pos_in_paragraph ASC"""
    )
    sql_query = sql_query.bindparams(sql.bindparam("articles_ids", expanding=True))
    all_sentences = pd.read_sql(sql_query, engine, params={"articles_ids": article_ids})

    groupby_var = all_sentences.groupby(by=["article_id", "paragraph_pos_in_article"])
    paragraphs = groupby_var["text"].apply(lambda x: " ".join(x))
    section_name = groupby_var["section_name"].unique().apply(lambda x: x[0])

    articles = pd.DataFrame(
        {"text": paragraphs, "section_name": section_name}
    ).reset_index()

    return articles


def retrieve_mining_cache(identifiers, etypes, engine):
    """Retrieve cached mining results.

    Parameters
    ----------
    identifiers : list of tuple
        Tuples of form (article_id, paragraph_pos_in_article). Note that if
        `paragraph_pos_in_article` is -1 then we are considering all the paragraphs.
    etypes : list
        List of entity types to consider. Duplicates are removed automatically.
    engine : sqlalchemy.engine.Engine
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    result : pd.DataFrame
        Selected rows of the `mining_cache` table.
    """
    logger = logging.getLogger("retrieve_mining_cache")
    logger.debug("parameters:")
    logger.debug(f"identifiers = {identifiers}")
    logger.debug(f"etypes = {etypes}")
    logger.debug(f"engine = {engine}")

    etypes = tuple(set(etypes))

    identifiers_arts = [int(a) for a, p in identifiers if p == -1]

    if identifiers_arts:
        query_arts = sql.text(
            """
        SELECT *
        FROM mining_cache
        WHERE article_id IN :identifiers_arts AND entity_type IN :etypes
        ORDER BY article_id, paragraph_pos_in_article, start_char
        """
        )
        query_arts = query_arts.bindparams(
            sql.bindparam("identifiers_arts", expanding=True),
            sql.bindparam("etypes", expanding=True),
        )
        df_arts = pd.read_sql(
            query_arts,
            con=engine,
            params={"identifiers_arts": identifiers_arts, "etypes": etypes},
        )
    else:
        logger.debug("setting df_arts to emtpy because `not identifiers_arts == True`")
        df_arts = pd.DataFrame()

    identifiers_pars = [(a, p) for a, p in identifiers if p != -1]
    if identifiers_pars:
        # Remarks
        # 1. Conditions are mutually exclusive, so several `UNION`s are
        #    equivalent to several `OR`s.
        # 2. `UNION` is considerably faster than `OR` in this case.
        # 3. If `len(identifiers_pars)` is too large, we may have a too long
        #    SQL statement which overflows the max length. So we break it down.

        if len(etypes) == 1:
            etypes = f"('{etypes[0]}')"
        batch_size = 1000
        dfs_pars = []
        d, r = divmod(len(identifiers_pars), batch_size)
        for i in range(0, d + (r > 0)):
            # Reformatted due to this bandit bug in python3.8:
            # https://github.com/PyCQA/bandit/issues/658
            query_pars = " UNION ".join(  # nosec
                "SELECT * FROM mining_cache "
                f"WHERE (article_id = {a} AND paragraph_pos_in_article = {p})"
                for a, p in identifiers_pars[i * batch_size : (i + 1) * batch_size]
            )
            # Reformatted due to this bandit bug in python3.8:
            # https://github.com/PyCQA/bandit/issues/658
            query_pars = (  # nosec
                f"SELECT * FROM ({query_pars}) tt " f"WHERE tt.entity_type IN {etypes}"
            )
            dfs_pars.append(pd.read_sql(query_pars, engine))
        df_pars = pd.concat(dfs_pars)
        # cast() to tell mypy that sort_values() doesn't return None here.
        df_pars = cast(
            pd.DataFrame,
            df_pars.sort_values(
                by=["article_id", "paragraph_pos_in_article", "start_char"]
            ),
        )
    else:
        logger.debug("setting df_pars to emtpy because `not identifiers_pars == True`")
        df_pars = pd.DataFrame()

    return df_pars.append(df_arts, ignore_index=True)


class SentenceFilter:
    """Filter sentence IDs by applying conditions.

    Instantiate this class and apply different filters by calling
    the corresponding filtering methods in any order. Finally,
    call either the `run()` or the `stream()` method to obtain
    the filtered sentence IDs.

    Example
    -------

    .. code-block:: python

        import sqlalchemy
        connection = sqlalchemy.create_engine("...")
        filtered_sentence_ids = (
            SentenceFilter(connection)
            .only_with_journal()
            .restrict_sentences_ids_to([1, 2, 3, 4, 5])
            .date_range((2010, 2020))
            .exclude_strings(["virus", "disease"])
            .run()
        )

    When the `run()` or the `stream()` method is called an SQL
    query is constructed and executed internally. For the example
    above it would have approximately the following form

    .. code-block:: SQL

        SELECT sentence_id
        FROM sentences
        WHERE
            article_id IN (
                SELECT article_id
                FROM articles
                WHERE
                    publish_time BETWEEN '2010-01-01' AND '2020-12-31' AND
                    journal IS NOT NULL
            ) AND
            sentence_id IN ('1', '2', '3', '4', '5') AND
            text NOT LIKE '%virus%' AND
            text NOT LIKE '%disease%'

    Parameters
    ----------
    connection : sqlalchemy.engine.Engine
        Connection to the database that contains the `articles`
        and `sentences` tables.
    """

    def __init__(self, connection):
        self.connection = connection
        self.logger = logging.getLogger(self.__class__.__name__)

        self.only_english_flag = False
        self.only_with_journal_flag = False
        self.discard_bad_sentences_flag = False
        self.year_from = None
        self.year_to = None
        self.string_exclusions = []
        self.string_inclusions = []
        self.restricted_sentence_ids = None

    def discard_bad_sentences(self, flag=True):
        """Discard sentences that are flagged as bad.

        Parameters
        ----------
        flag : bool
            If True, then all sentences with `True` in the `is_bad`
            column are discarded.

        Returns
        -------
        self : SentenceFilter
            The instance of `SentenceFilter` itself. Useful for
            chained applications of filters.
        """
        self.logger.info(f"Discard bad: {flag}")
        self.discard_bad_sentences_flag = flag
        return self

    def only_english(self, flag=True):
        """Only select articles that are in English.

        Parameters
        ----------
        flag : bool
            If True, then only articles for which are in English
            will be selected.

        Returns
        -------
        self : SentenceFilter
            The instance of `SentenceFilter` itself. Useful for
            chained applications of filters.
        """
        self.logger.info(f"Only in English: {flag}")
        self.only_english_flag = flag
        return self

    def only_with_journal(self, flag=True):
        """Only select articles with a journal.

        Parameters
        ----------
        flag : bool
            If True, then only articles for which a journal was
            specified will be selected.

        Returns
        -------
        self : SentenceFilter
            The instance of `SentenceFilter` itself. Useful for
            chained applications of filters.
        """
        self.logger.info(f"Only with journal: {flag}")
        self.only_with_journal_flag = flag
        return self

    def date_range(self, date_range=None):
        """Restrict to articles in a given date range.

        Parameters
        ----------
        date_range : tuple or None
            A tuple with two elements of the form `(start_year, end_year)`.
            If None then nothing no date range is applied.

        Returns
        -------
        self : SentenceFilter
            The instance of `SentenceFilter` itself. Useful for
            chained applications of filters.
        """
        self.logger.info(f"Date range: {date_range}")
        if date_range is not None:
            self.year_from, self.year_to = date_range
        return self

    def include_strings(self, strings):
        """Include only sentences containing all of the given strings.

        Parameters
        ----------
        strings : list_like
            The strings to include.

        Returns
        -------
        self : SentenceFilter
            The instance of `SentenceFilter` itself.
        """
        self.logger.info(f"Include strings: {strings}")
        strings = map(lambda s: s.lower(), strings)
        strings = filter(lambda s: len(s) > 0, strings)
        self.string_inclusions.extend(strings)
        return self

    def exclude_strings(self, strings):
        """Exclude sentences containing any of the given strings.

        Parameters
        ----------
        strings : list_like
            The strings to exclude.

        Returns
        -------
        self : SentenceFilter
            The instance of `SentenceFilter` itself. Useful for
            chained applications of filters.
        """
        self.logger.info(f"Exclude strings: {strings}")
        strings = map(lambda s: s.lower(), strings)
        strings = filter(lambda s: len(s) > 0, strings)
        self.string_exclusions.extend(strings)
        return self

    def restrict_sentences_ids_to(self, sentence_ids):
        """Restrict sentence IDs to the given ones.

        Parameters
        ----------
        sentence_ids : list_like
            The sentence IDs to restrict to.

        Returns
        -------
        self : SentenceFilter
            The instance of `SentenceFilter` itself. Useful for
            chained applications of filters.
        """
        # For logging
        if len(sentence_ids) > 5:
            ids_str = f"[{', '.join(map(str, sentence_ids[:5]))} ..."
        else:
            ids_str = str(sentence_ids)
        self.logger.info(f"Restricting to sentencs IDs: {ids_str}")

        # The actual restriction
        self.restricted_sentence_ids = tuple(sentence_ids)

        return self

    def _build_query(self):
        article_conditions = []
        sentence_conditions = []

        # Discard bad condition
        if self.discard_bad_sentences_flag:
            sentence_conditions.append("is_bad = 0")

        # In English condition
        if self.only_english_flag:
            article_conditions.append("is_english = 1")

        # Journal condition
        if self.only_with_journal_flag:
            article_conditions.append("journal IS NOT NULL")

        # Date range condition
        if self.year_from is not None and self.year_to is not None:
            from_date = f"{self.year_from:04d}-01-01"
            to_date = f"{self.year_to:04d}-12-31"
            article_conditions.append(
                f"publish_time BETWEEN '{from_date}' AND '{to_date}'"
            )

        # Add article conditions to sentence conditions
        if len(article_conditions) > 0:
            # Reformatted due to this bandit bug in python3.8:
            # https://github.com/PyCQA/bandit/issues/658
            article_condition_query = (  # nosec
                "article_id IN ( "
                "    SELECT article_id "
                "    FROM articles "
                f'    WHERE {" AND ".join(article_conditions)} '
                ")"
            ).strip()  # nosec
            sentence_conditions.append(article_condition_query)

        # Restricted sentence IDs
        if self.restricted_sentence_ids is not None:
            sentence_ids_s = ", ".join(str(x) for x in self.restricted_sentence_ids)
            if not sentence_ids_s and self.connection.url.drivername in {
                "mysql+mysqldb",
                "mysql+pymysql",
            }:
                sentence_ids_s = "NULL"
            sentence_conditions.append(f"sentence_id IN ({sentence_ids_s})")

        # Inclusion and Exclusion Text
        if self.connection.url.drivername in {"mysql+mysqldb", "mysql+pymysql"}:
            if self.string_inclusions:
                inclusions = " ".join(
                    f'+"{string}"' if len(string.split(" ")) > 1 else f"+{string}"
                    for string in self.string_inclusions
                )
                exclusions = " ".join(
                    f'-"{string}"' if len(string.split(" ")) > 1 else f"-{string}"
                    for string in self.string_exclusions
                )
                condition = f"{inclusions} {exclusions}".strip()
                sentence_conditions.append(
                    f"MATCH(text) AGAINST ('{condition}' IN BOOLEAN MODE)"
                )
            elif self.string_exclusions:
                # This elif statement is to create conditions if there are
                # onlyexclusions words; without any inclusions. Indeed,
                # in this case, MATCH AGAINST IN BOOLEAN MODE does; not work
                # anymore as you can find on the official docs:
                # https://dev.mysql.com/doc/refman/8.0/en/fulltext-boolean.html
                # boolean-mode search that contains only terms preceded by -
                # returns an empty result
                for text in self.string_exclusions:
                    sentence_conditions.append(f"INSTR(text, '{text}') = 0")
        else:
            for text in self.string_exclusions:
                sentence_conditions.append(f"text NOT LIKE '%{text}%'")
            for text in self.string_inclusions:
                sentence_conditions.append(f"text LIKE '%{text}%'")

        # Build and send query
        query = "SELECT sentence_id FROM sentences"
        if len(sentence_conditions) > 0:
            query = f"{query} WHERE {' AND '.join(sentence_conditions)}"

        return query

    def iterate(self, chunk_size):
        """Run the filtering query and iterate over restricted sentence IDs.

        Parameters
        ----------
        chunk_size : int
            The size of the batches of sentence IDs that are yielded.

        Yields
        ------
        result_arr : np.ndarray
            A 1-dimensional numpy array with the filtered sentence IDs.
            Its length will be at most equal to `chunk_size`.
        """
        self.logger.info(f"Iterating filtering with chunk size {chunk_size}")

        query = self._build_query()
        # self.logger.info(f"Query: {query}")
        for df_results in pd.read_sql(query, self.connection, chunksize=chunk_size):
            result_arr = df_results["sentence_id"].to_numpy()
            yield result_arr

    def run(self):
        """Run the filtering query to find restricted sentence IDs.

        Returns
        -------
        result_arr : np.ndarray
            A 1-dimensional numpy array with the filtered sentence IDs.
        """
        self.logger.info("Running the filtering query")

        query = self._build_query()
        # self.logger.info(f"Query: {query}")

        self.logger.debug("Running pd.read_sql")
        results = [row[0] for row in self.connection.execute(query).fetchall()]

        self.logger.info(f"Filtering gave {len(results)} results")

        return np.array(results)
