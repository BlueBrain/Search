"""
SQL Related functions.

whatever
"""
import pandas as pd
from sqlalchemy import Column, String, Integer, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Articles(Base):
    __tablename__ = 'articles'
    __table_args__ = {'extend_existing': True}

    article_id = Column(String, primary_key=True)
    publisher = Column(String)
    title = Column(String)
    doi = Column(String)
    pmc_id = Column(String)
    pm_id = Column(Integer)
    licence = Column(String)
    abstract = Column(String)
    date = Column(String)
    authors = Column(String)
    journal = Column(String)
    microsoft_id = Column(Integer)
    covidence_id = Column(String)
    has_pdf_parse = Column(Boolean)
    has_pmc_xml_parse = Column(Boolean)
    has_covid19_tag = Column(Boolean)
    fulltext_directory = Column(String)
    url = Column(String)

    def init(self, article_id, publisher, title, doi, pmc_id, pm_id, licence, abstract, date, authors, journal,
             microsoft_id,
             covidence_id, has_pdf_parse, has_pmc_xml_parse, has_covid19_tag, fulltext_directory, url):
        self.article_id = article_id
        self.publisher = publisher
        self.title = title
        self.doi = doi
        self.pmc_id = pmc_id
        self.pm_id = pm_id
        self.license = licence
        self.abstract = abstract
        self.date = date
        self.authors = authors
        self.journal = journal
        self.microsoft_id = microsoft_id
        self.covidence_id = covidence_id
        self.has_pdf_parse = has_pdf_parse
        self.has_pmc_xml_parse = has_pmc_xml_parse
        self.has_covid19_tag = has_covid19_tag
        self.fulltext_directory = fulltext_directory
        self.url = url


class Article_id_2_sha(Base):
    __tablename__ = 'article_id_2_sha'
    __table_args__ = {'extend_existing': True}

    article_id = Column(String, ForeignKey('articles.article_id'), primary_key=True)
    sha = Column(String)

    def init(self, article_id, sha):
        self.article_id = article_id
        self.sha = sha


class Sentences(Base):
    __tablename__ = 'sentences'
    __table_args__ = {'extend_existing': True}

    sentence_id = Column(Integer, primary_key=True)
    sha = Column(String, ForeignKey('article_id_2_sha.sha'))
    section_name = Column(String)
    text = Column(String)
    paragraph_id = Column(Integer)

    def init(self, sentence_id, sha, section_name, text, paragraph_id):
        self.sentence_id = sentence_id
        self.sha = sha
        self.section_name = section_name
        self.text = text
        self.paragraph_id = paragraph_id


class Paragraphs(Base):
    __tablename__ = 'paragraphs'
    __table_args__ = {'extend_existing': True}

    paragraph_id = Column(Integer, primary_key=True)
    sha = Column(String, ForeignKey('article_id_2_sha.sha'))
    section_name = Column(String)
    text = Column(String)

    def init(self, paragraph_id, sha, section_name, text):
        self.paragraph_id = paragraph_id
        self.sha = sha
        self.section_name = section_name
        self.text = text


def get_paragraph_ids(article_ids, db_cnxn):
    """Given a list of article ids find all the corresponding paragraph ids.

    Parameters
    ----------
    article_ids : list
        List of article ids. Note that they are the primary keys in the `articles` table.

    db_cnxn : sqlite3.Connection
        Connection to the database.

    Returns
    -------
    pd.Series
        The unique index represents the paragraph ids and the values represent the article ids.
    """
    article_ids_joined = ','.join(f"\"{id_}\"" for id_ in set(article_ids))

    sql_query = f"""
    SELECT article_id, paragraph_id
    FROM (
             SELECT paragraph_id, sha
             FROM paragraphs
             WHERE sha IN (
                 SELECT sha
                 FROM article_id_2_sha
                 WHERE article_id IN ({article_ids_joined})
             )
         ) p
             INNER JOIN
         article_id_2_sha a
         ON a.sha = p.sha;
    """
    results = pd.read_sql(sql_query, db_cnxn)

    return pd.Series(results['article_id'], index=results['paragraph_id'])


def find_paragraph(sentence_id, session):
    """Find the paragraph corresponding to the given sentence.

    Parameters
    ----------
    sentence_id : int
        The identifier of the given sentence
    session: SQLAlchemy.orm.session
        Cursor to the database

    Returns
    -------
    paragraph : str
        The paragraph containing `sentence`
    """
    sentence = session.query(Sentences).filter(Sentences.sentence_id == sentence_id).one()
    paragraph_id = sentence.paragraph_id
    paragraph = session.query(Paragraphs).filter(Paragraphs.paragraph_id == paragraph_id).one()

    return paragraph.text


def get_shas_from_ids(articles_ids, db):
    """Find articles SHA given article IDs.

    Parameters
    ----------
    articles_ids : list
        A list of strings representing article IDs.
    db : sqlite3.Cursor
        A SQL database for querying the SHAs. Should contain
        a table named "article_id_2_sha".

    Returns
    -------
    results : list
        A list of sentence SHAs.
    """
    all_ids_str = ', '.join([f"'{id_}'" for id_ in articles_ids])
    sql_query = f"SELECT sha FROM article_id_2_sha WHERE article_id IN ({all_ids_str})"
    results = db.execute(sql_query).fetchall()
    results = [sha for (sha,) in results]

    return results


def get_ids_by_condition(conditions, table, db):
    """Find entry IDs given a number of search conditions.

    Notes
    -----
    In the database 'cord19', tables are named with plural noun (e.g sentences, articles)
    However, column id are named in singular form (e.g. sentence, article)

    Parameters
    ----------
    conditions : list
        A list strings representing SQL query conditions. They should be
        formatted so that they can be used in an SQL WHERE statement,
        for example:
            SELECT * FROM {table} WHERE <condition_1> and <condition_2>"
    table : str
        The name of the table in `db`.
    db : sqlite3.Cursor
        A SQL database for querying the article IDs. Should contain
        a table named "articles".

    Returns
    -------
    results : list
        A list of article IDs (=SHAs) represented as strings.
    """
    if conditions:
        condition = ' and '.join(conditions)
        sql_query = f"SELECT {table[:-1]}_id FROM {table} WHERE {condition}"
    else:
        sql_query = f"SELECT {table[:-1]}_id FROM {table}"
    results = db.execute(sql_query).fetchall()
    results = [id_ for (id_,) in results]

    return results


class ArticleConditioner:
    """Article conditioner."""

    @staticmethod
    def get_date_range_condition(date_range):
        """Construct a date range condition.

        Notes
        -----
        Rows without a date entry will never be selected (to be checked)

        Parameters
        ----------
        date_range : tuple
            A tuple of the form (date_from, date_to)

        Returns
        -------
        condition : str
            The SQL condition
        """
        date_from, date_to = date_range
        condition = f"date BETWEEN '{date_from}-01-01' and '{date_to}-12-31'"

        return condition

    @staticmethod
    def get_has_journal_condition():
        """Construct a has-journal condition.

        Returns
        -------
        condition : str
            The SQL condition
        """
        condition = "journal IS NOT NULL"

        return condition

    @staticmethod
    def get_restrict_to_tag_condition(tag='has_covid19_tag'):
        """Construct a condition for restricting to a given tag.

        Parameters
        ----------
        tag : str
            The tag to constrain on

        Returns
        -------
        condition : str
            The SQL condition
        """
        condition = f"{tag} = 1"
        return condition


class SentenceConditioner:
    """Sentence conditioner."""

    @staticmethod
    def get_word_exclusion_condition(word):
        """Construct condition for exclusion containing a given word.

        Parameters
        ----------
        word

        Returns
        -------
        condition
        """
        condition = f"text NOT LIKE '%{word}%'"
        return condition
