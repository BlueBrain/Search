"""Module for the Database Creation."""
import json
import pandas as pd
from pathlib import Path
import re
# import sqlite3

import spacy
import sqlalchemy
from sqlalchemy import Column, String, Integer, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

class Articles(Base):
    """Articles table."""
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
        """Init of the articles table."""
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
    """Article_id_2_sha table."""
    __tablename__ = 'article_id_2_sha'
    __table_args__ = {'extend_existing': True}
    article_id = Column(String, ForeignKey('articles.article_id'), primary_key=True)
    sha = Column(String)
    def init(self, article_id, sha):
        """Init of the article_id_2_sha table."""
        self.article_id = article_id
        self.sha = sha
class Sentences(Base):
    """Sentences table."""
    __tablename__ = 'sentences'
    __table_args__ = {'extend_existing': True}
    sentence_id = Column(Integer, primary_key=True)
    sha = Column(String, ForeignKey('article_id_2_sha.sha'))
    section_name = Column(String)
    text = Column(String)
    paragraph_id = Column(Integer)
    def init(self, sentence_id, sha, section_name, text, paragraph_id):
        """Init of the sentences table."""
        self.sentence_id = sentence_id
        self.sha = sha
        self.section_name = section_name
        self.text = text
        self.paragraph_id = paragraph_id
class Paragraphs(Base):
    """Paragraphs table."""
    __tablename__ = 'paragraphs'
    __table_args__ = {'extend_existing': True}
    paragraph_id = Column(Integer, primary_key=True)
    sha = Column(String, ForeignKey('article_id_2_sha.sha'))
    section_name = Column(String)
    text = Column(String)
    def init(self, paragraph_id, sha, section_name, text):
        """Init of the paragraphs table."""
        self.paragraph_id = paragraph_id
        self.sha = sha
        self.section_name = section_name
        self.text = text


class CORD19DatabaseCreation:
    """Create SQL database from a specified dataset."""

    def __init__(self,
                 data_path,
                 version,
                 saving_directory=None):
        """Create SQL database object.

        Parameters
        ----------
        data_path: pathlib.Path
            Directory to the dataset where metadata.csv and all jsons file are located.
        version: str
            Version of the database created.
        saving_directory: pathlib.Path
            Directory where the database is going to be saved.
        """
        self.data_path = data_path
        if not self.data_path.exists():
            raise NotADirectoryError(f'The data directory {self.data_path} does not exit')

        self.version = version

        self.saving_directory = saving_directory or Path.cwd()
        if not self.saving_directory.exists():
            raise NotADirectoryError(f'The saving directory {self.saving_directory} does not exit')

        self.filename = self.saving_directory / f'cord19_{self.version}.db'
        if self.filename.exists():
            raise ValueError(f'The version {self.version} of the database already exists')

        self.metadata = pd.read_csv(self.data_path / 'metadata.csv')
        self.is_constructed = False
        # self.db = sqlite3.connect(str(self.filename))
        self.engine = sqlalchemy.create_engine(f'sqlite:///{self.filename}')
        # self.all_json_paths = self.data_path.rglob("*.json")

    def construct(self):
        """Construct the database."""
        if not self.is_constructed:
            self._schema_creation()
            print('Schemas of the tables are created.')
            self._articles_table()
            print('Articles table is created.')
            self._sentences_table()
            print('Sentences table is created.')
            self.is_constructed = True
        else:
            raise ValueError('This database is already constructed!')

    def _schema_creation(self):
        """Create the schemas of the different tables in the database."""
        metadata = sqlalchemy.MetaData()

        self.articles_table = \
            sqlalchemy.Table('articles', metadata,
                             sqlalchemy.Column('article_id', sqlalchemy.Integer(),
                                               primary_key=True, autoincrement=True),
                             sqlalchemy.Column('cord_uid', sqlalchemy.String(8), nullable=False),
                             sqlalchemy.Column('sha', sqlalchemy.String(40)),
                             sqlalchemy.Column('source_x', sqlalchemy.Text()),
                             sqlalchemy.Column('title', sqlalchemy.Text()),
                             sqlalchemy.Column('doi', sqlalchemy.Text()),
                             sqlalchemy.Column('pmcid', sqlalchemy.Text()),
                             sqlalchemy.Column('pubmed_id', sqlalchemy.Text()),
                             sqlalchemy.Column('license', sqlalchemy.Text()),
                             sqlalchemy.Column('abstract', sqlalchemy.Text()),
                             sqlalchemy.Column('publish_time', sqlalchemy.Date()),
                             sqlalchemy.Column('authors', sqlalchemy.Text()),
                             sqlalchemy.Column('journal', sqlalchemy.Text()),
                             sqlalchemy.Column('mag_id', sqlalchemy.Text()),
                             sqlalchemy.Column('who_covidence_id', sqlalchemy.Text()),
                             sqlalchemy.Column('arxiv_id', sqlalchemy.Text()),
                             sqlalchemy.Column('pdf_json_files', sqlalchemy.Text()),
                             sqlalchemy.Column('pmc_json_files', sqlalchemy.Text()),
                             sqlalchemy.Column('url', sqlalchemy.Text()),
                             sqlalchemy.Column('s2_id', sqlalchemy.Text())
                             )

        self.sentences_table = \
            sqlalchemy.Table('sentences', metadata,
                             sqlalchemy.Column('sentence_id', sqlalchemy.Integer(),
                                               primary_key=True, autoincrement=True),
                             sqlalchemy.Column('section_name', sqlalchemy.Text()),
                             sqlalchemy.Column('article_id', sqlalchemy.Integer(),
                                               sqlalchemy.ForeignKey("articles.article_id"),
                                               nullable=False),
                             sqlalchemy.Column('text', sqlalchemy.Text()),
                             sqlalchemy.Column('paragraph_pos_in_article', sqlalchemy.Integer(),
                                               nullable=False),
                             sqlalchemy.Column('sentence_pos_in_paragraph', sqlalchemy.Integer(),
                                               nullable=False)
                             )

        with self.engine.begin() as connection:
            metadata.create_all(connection)

    def _articles_table(self):
        """Fill the Article Table thanks to 'metadata.csv'.

        The articles table has all the metadata.csv columns
        expect the 'sha'.
        Moreover, the columns are renamed (cfr. _rename_columns)
        """
        df = self.metadata.copy()
        df.drop_duplicates('cord_uid', keep='first', inplace=True)
        df.to_sql(name='articles', con=self.engine, index=False, if_exists='append')

    def _sentences_table(self, model_name='en_core_sci_lg'):
        """Fill the sentences table thanks to all the json files.

        For each paragraph, all sentences are extracted and populate
        the sentences table.

        Parameters
        ----------
        model_name: str
            SpaCy model used to parse the text into sentences.

        Returns
        -------
        pmc: int
            Number of articles with at least one pmc_json.
        pdf: int
            Number of articles that does not have pmc_json file but at least one pdf_json.
        """
        nlp = spacy.load(model_name, disable=["tagger", "ner"])

        articles_table = pd.read_sql("""SELECT article_id, title, abstract, pmc_json_files, pdf_json_files 
                                        FROM articles
                                        WHERE (abstract IS NOT NULL) OR (title IS NOT NULL)""",
                                     con=self.engine)

        pdf = 0
        pmc = 0
        for _, article in articles_table.iterrows():
            sentences = []
            article_id = int(article['article_id'])
            paragraph_pos_in_article = 0

            # Read title and abstract
            if article['title'] is not None:
                sentences += [{
                    'section_name': 'Title',
                    'article_id': article_id,
                    'text': sent,
                    'paragraph_pos_in_article': paragraph_pos_in_article,
                    'sentence_pos_in_paragraph': sentence_pos_in_paragraph,
                } for sentence_pos_in_paragraph, sent
                    in enumerate(self.segment(nlp, article['title']))]
                paragraph_pos_in_article += 1
            if article['abstract'] is not None:
                sentences += [{
                    'section_name': 'Abstract',
                    'article_id': article_id,
                    'text': sent,
                    'paragraph_pos_in_article': paragraph_pos_in_article,
                    'sentence_pos_in_paragraph': sentence_pos_in_paragraph,
                } for sentence_pos_in_paragraph, sent in enumerate(self.segment(nlp, article['abstract']))]
                paragraph_pos_in_article += 1

            # Find files linked to articles
            if article['pmc_json_files'] is not None:
                pmc += 1
                jsons_path = article['pmc_json_files'].split('; ')
            elif article['pdf_json_files'] is not None:
                pdf += 1
                jsons_path = article['pdf_json_files'].split('; ')
            else:
                jsons_path = []

            # Load json
            for json_path in jsons_path:
                json_path = self.data_path + json_path.strip()
                with open(str(json_path), 'r') as json_file:
                    file = json.load(json_file)

                    for paragraph_pos_in_article, section in enumerate(file['body_text'],
                                                                       start=paragraph_pos_in_article):
                        sentences += [{
                            'section_name': section['section'].title(),
                            'article_id': article_id,
                            'text': sent,
                            'paragraph_pos_in_article': paragraph_pos_in_article,
                            'sentence_pos_in_paragraph': sentence_pos_in_paragraph,
                        } for sentence_pos_in_paragraph, sent in enumerate(self.segment(nlp, section['text']))]

                    for paragraph_pos_in_article, (_, v) in enumerate(file['ref_entries'].items(),
                                                                      start=paragraph_pos_in_article):
                        sentences += [{
                            'section_name': 'Caption',
                            'article_id': article_id,
                            'text': sent,
                            'paragraph_pos_in_article': paragraph_pos_in_article,
                            'sentence_pos_in_paragraph': sentence_pos_in_paragraph,
                        } for sentence_pos_in_paragraph, sent in enumerate(self.segment(nlp, v['text']))]

            sentences_df = pd.DataFrame(sentences, columns=['sentence_id', 'section_name', 'article_id',
                                                            'text', 'paragraph_pos_in_article',
                                                            'sentence_pos_in_paragraph'])
            sentences_df.to_sql(name='sentences', con=self.engine, index=False, if_exists='append')

        return pmc, pdf

    @staticmethod
    def segment(nlp, paragraph):
        """Segment a paragraph/article into sentences.

        Parameters
        ----------
        nlp: spacy.language.Language
            Spacy pipeline applying sentence segmentation.
        paragraph: str
            Paragraph/Article in raw text to segment into sentences.

        Returns
        -------
        all_sentences: list
            List of all the sentences extracted from the paragraph.
        """
        all_sentences = [sent.string.strip() for sent in nlp(paragraph).sents]
        return all_sentences
