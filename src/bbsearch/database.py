"""Module for the Database Creation."""
import json
import pandas as pd
from pathlib import Path
import re
# import sqlite3

import sqlalchemy
from sqlalchemy import Column, String, Integer, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

from spacy.lang.en import English
from spacy.attrs import ORTH, LEMMA

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
            # self._rename_columns()
            self._schema_creation()
            print('Schemas of the tables are created.')
            # self._article_id_to_sha_table()
            print('Article_id_2_sha table is created.')
            self._articles_table()
            print('Articles table is created.')
            self._paragraphs_table()
            print('Paragraphs table is created.')
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

    # def _rename_columns(self):
    #     """Rename the columns of the dataframe to follow the SQL database schema."""
    #     self.metadata.rename(columns={
    #         'cord_uid': 'article_id',
    #         'sha': 'sha',
    #         'source_x': 'publisher',
    #         'title': 'title',
    #         'doi': 'doi',
    #         'pmcid': 'pmc_id',
    #         'pubmed_id': 'pm_id',
    #         'license': 'licence',
    #         'abstract': 'abstract',
    #         'publish_time': 'date',
    #         'authors': 'authors',
    #         'journal': 'journal',
    #         'Microsoft Academic Paper ID': 'microsoft_id',
    #         'WHO #Covidence': 'covidence_id',
    #         'has_pdf_parse': 'has_pdf_parse',
    #         'has_pmc_xml_parse': 'has_pmc_xml_parse',
    #         'full_text_file': 'fulltext_directory',
    #         'url': 'url'}, inplace=True)

    def _articles_table(self):
        """Fill the Article Table thanks to 'metadata.csv'.

        The articles table has all the metadata.csv columns
        expect the 'sha'.
        Moreover, the columns are renamed (cfr. _rename_columns)
        """
        df = self.metadata.copy()
        df.drop_duplicates('cord_uid', keep='first', inplace=True)
        df.to_sql(name='articles', con=self.engine, index=False, if_exists='append')

    # def _article_id_to_sha_table(self):
    #     """Fill the article_id_to_sha table thanks to 'metadata.csv'.
    #
    #     The metadata.csv columns called 'sha' and 'article_id'
    #     are used to create this table.
    #     Moreover, several article_id can refer to the same shas
    #     (for example, supplementary material, ...).
    #     Several shas can also refer to the same article_id.
    #     """
    #     df = self.metadata[['article_id', 'sha']]
    #     df = df.set_index(['article_id']).apply(lambda x: x.str.split('; ').explode()).reset_index()
    #     df.to_sql(name='article_id_2_sha', con=self.db, index=False, if_exists='append')

    # def _paragraphs_table(self):
    #     """Fill the paragraphs table thanks to all the json files.
    #
    #     For each article_id, this function extracts the corresponding
    #     'sha' (and thus the json files) and extracts body_text and ref_entries
    #     from those json_files.
    #     """
    #     sha_to_json_path = {json_path.stem: json_path for json_path in self.all_json_paths}
    #     cur = self.sqlalchemy.cursor()
    #     for (article_id,) in cur.execute('SELECT article_id FROM articles'):
    #         tag, paragraphs = self.get_tag_and_paragraph(sha_to_json_path, article_id)
    #         self.update_covid19_tag(article_id, tag)
    #         self.insert_into_paragraphs(paragraphs)
    #     self.sqlalchemy.commit()

    def _sentences_table(self):
        """Fill the sentences table thanks to all the json files.

        For each paragraph, all sentences are extracted and populate
        the sentences table.
        """
        nlp = self.define_nlp()
        cur = self.sqlalchemy.cursor()
        for (paragraph_id,) in cur.execute('SELECT paragraph_id FROM paragraphs'):
            sentences = self.get_sentences(nlp, paragraph_id)
            self.insert_into_sentences(sentences)
        self.sqlalchemy.commit()

    def define_nlp(self):
        """Create the sentence boundary detection tools from Spacy.

        Notes
        -----
        Some custom abbreviations are added to the basic Spacy tool.

        Returns
        -------
        nlp: spacy.lang.en
            SBD tool from Spacy with custom abbreviations.
        """
        nlp = English()
        sbd = nlp.create_pipe('sentencizer')
        nlp.add_pipe(sbd)
        self.add_abbreviations(nlp)

        return nlp

    def get_tag_and_paragraph(self, sha_to_json_path, article_id):
        """Extract paragraphs from given article and identify if is about covid19.

        Notes
        -----
        Here are the steps implemented:
        - First, we look at the corresponding shas for the given article_id
        - Paragraphs are extracted:
            - From Title and Abstract in the articles table
            - From body_text and ref_entries in the json files.
        - Tag is a boolean to identify if article speaks about COVID19.

        Parameters
        ----------
        sha_to_json_path: dict
            Dictionary where key values are  shas.
        article_id: str
            ID of the article specified in the articles database.

        Returns
        -------
        tag: boolean
            Tag value of has_covid19. This is checking if covid19 is mentioned in the paper.
        paragraphs: list
            List of the extracted paragraphs. (paragraph_id, paragraph)
        """
        paragraphs = []
        tag = False

        article_id, article_title, article_abstract, article_directory = self.sqlalchemy.execute(
            "SELECT article_id, title, abstract, fulltext_directory FROM articles WHERE article_id is ?",
            [article_id]).fetchone()

        all_shas = self.sqlalchemy.execute("SELECT sha FROM article_id_2_sha WHERE article_id = ?", [article_id]).fetchall()
        title_sha = all_shas[0][0] if all_shas else None
        if article_title:
            paragraphs.append((title_sha, 'Title', article_title))
        if article_abstract:
            paragraphs.append((title_sha, 'Abstract', article_abstract))

        for (sha,) in all_shas:
            if sha:
                if sha in sha_to_json_path.keys():
                    found_json_files = sha_to_json_path[sha]
                else:
                    raise ValueError(f'Did not found json files for sha {sha}')
                with open(str(found_json_files)) as json_file:
                    file = json.load(json_file)
                    for sec in file['body_text']:
                        paragraphs.append((sha, sec['section'].title(), sec['text']))
                    for _, v in file['ref_entries'].items():
                        paragraphs.append((sha, 'Caption', v['text']))

        tag = tag or self.get_tags(paragraphs)

        return tag, paragraphs

    # def get_sentences(self, nlp, paragraph_id):
    #     """Extract all the sentences from the paragraph table.
    #
    #     Parameters
    #     ----------
    #     nlp: spacy.language.Language
    #         Sentence Boundary Detection tool from Spacy to seperate sentences.
    #     paragraph_id: int
    #         ID of the paragraph
    #
    #     Returns
    #     -------
    #     sentences: list
    #         List of the extracted sentences.
    #     """
    #     sentences = []
    #     sha, section_name, paragraph = self.sqlalchemy.execute(
    #         "SELECT sha, section_name, text FROM paragraphs WHERE paragraph_id = ?",
    #         [paragraph_id]).fetchall()[0]
    #     sentences += [(sha, section_name, sent, paragraph_id) for sent in self.segment(nlp, paragraph)]
    #
    #     return sentences

    def update_covid19_tag(self, article_id, tag):
        """Update the covid19 tag in the articles database.

        Parameters
        ----------
        article_id: str
            Article ID of the row to update into the database.
        tag: boolean
            Value of the tag. True if covid19 is mentionned, otherwise False.
        """
        self.sqlalchemy.execute("UPDATE articles SET has_covid19_tag = ? WHERE article_id = ?", [tag, article_id])

    def insert_into_sentences(self, sentences):
        """Insert the new sentences into the database sentences.

        Parameters
        ----------
        sentences: list
            List of sentences to insert in format (sha, section_name, text, paragraph_id)
        """
        cur = self.sqlalchemy.cursor()
        cur.executemany("INSERT INTO sentences (sha, section_name, text, paragraph_id) VALUES (?, ?, ?, ?)", sentences)

    def insert_into_paragraphs(self, paragraphs):
        """Insert the new sentences into the database sentences.

        Parameters
        ----------
        paragraphs: list
            List of sentences to insert in format (paragraph_id, text)
        """
        cur = self.sqlalchemy.cursor()
        cur.executemany("INSERT INTO paragraphs (sha, section_name, text) VALUES (?, ?, ?)", paragraphs)

    @staticmethod
    def get_tags(sentences):
        """Compute the tag for an article id through its sentences.

        Notes
        -----
        This tag is used to filter articles that contains mentions to covid19.
        The list of words is:
        'covid', 'covid 19', 'covid-19',
        'sars cov 2', 'sars-cov 2',
        '2019 ncov', '2019ncov', '2019-ncov', '2019 n cov', '2019n cov',
        '2019 novel coronavirus',  'coronavirus 2019',
        'cov-2019', 'cov 2019',
        'coronavirus disease 2019', 'coronavirus disease 19', 'coronavirus disease'
        'wuhan coronavirus', 'wuhan cov', 'wuhan pneumonia'

        Parameters
        ----------
        sentences: list
            List of sentences from an article_id in the format (sha, name, text)

        Returns
        -------
        tag: boolean
            Value of the tag has_covid19 of the corresponding article_id
        """
        # Keyword patterns to search for
        keywords = [r"2019[\-\s]?n[\-\s]?cov", "2019 novel coronavirus",
                    "coronavirus 2019", r"coronavirus disease (?:20)?19",
                    r"covid(?:[\-\s]?19)?", r"n\s?cov[\-\s]?2019", r"sars-cov-?2",
                    r"wuhan (?:coronavirus|cov|pneumonia)"]
        # Build regular expression for each keyword. Wrap term in word boundaries
        regex = "|".join([f"\\b{keyword}\\b" for keyword in keywords])
        tag = False  # None
        for _, _, text in sentences:
            # Look for at least one keyword match
            if re.findall(regex, text.lower()):
                tag = True  # "COVID-19"
                break
        return tag

    @staticmethod
    def remove_sentences_duplicates(sentences):
        """Return a filtered list of sentences.

        Notes
        -----
        Duplicate and boilerplate text strings are removed.
        This is done to avoid duplicates coming from metadata.csv and raw json files.

        Parameters
        ----------
        sentences: list
            List of sentences with format (sha, name, text) from an article_id

        Returns
        -------
        unique: list
            List of sentences (without duplicates) with format (sha, name, text)
        """
        # Use list to preserve insertion order
        unique = []
        seen = set()

        # Boilerplate text to ignore
        boilerplate = {"COVID-19 resource centre",
                       "permission to make all its COVID",
                       "WHO COVID database"}

        for sha, name, text in sentences:
            # Add unique text that isn't boilerplate text
            if text not in seen and not any(x in text for x in boilerplate):
                unique.append((sha, name, text))
                seen.add(text)

        return unique

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

    @staticmethod
    def add_abbreviations(nlp, abbreviations=None):
        """Add new abbreviations to the default list to avoid wrong scission.

        Parameters
        ----------
        nlp : spacy.lang.en.English()
            Spacy NLP used for the sentence boundary detection.
        abbreviations: list of tuples
            New abbreviations to add to the default list.
            Format: (abbreviation, [{ORTH: value, LEMMA: value}])
        """
        default_abbreviations = [('approx.', [{ORTH: 'approximately', LEMMA: 'approximately'}]),
                                 ('cf.', [{ORTH: 'cf.', LEMMA: 'confer'}]),
                                 ('et al.', [{ORTH: 'et al.', LEMMA: 'and others'}]),
                                 ('Fig.', [{ORTH: 'Figure', LEMMA: 'figure'}]),
                                 ('fig.', [{ORTH: 'figure', LEMMA: 'figure'}]),
                                 ('Figs.', [{ORTH: 'figures', LEMMA: 'figures'}]),
                                 ('Eqs.', [{ORTH: 'Equations', LEMMA: 'equations'}]),
                                 ('Eq.', [{ORTH: 'Equation', LEMMA: 'equation'}]),
                                 ('Sec.', [{ORTH: 'Section', LEMMA: 'section'}]),
                                 ('Ref.', [{ORTH: 'References', LEMMA: 'references'}]),
                                 ('App.', [{ORTH: 'Appendix', LEMMA: 'appendix'}]),
                                 ('Nat.', [{ORTH: 'Natural', LEMMA: 'natural'}]),
                                 ('min.', [{ORTH: 'Minimum', LEMMA: 'minimum'}]),
                                 ('etc.', [{ORTH: 'etc.', LEMMA: 'Et Cetera'}]),
                                 ('Sci.', [{ORTH: 'Scientific', LEMMA: 'figure'}]),
                                 ('Proc.', [{ORTH: 'Proceedings', LEMMA: 'proceedings'}]),
                                 ('Acad.', [{ORTH: 'Academy', LEMMA: 'Academy'}]),
                                 ('No.', [{ORTH: 'Number', LEMMA: 'Number'}]),
                                 ('Med.', [{ORTH: 'Medecine', LEMMA: 'medecine'}]),
                                 ('Rev.', [{ORTH: 'Review', LEMMA: 'review'}]),
                                 ('Subsp.', [{ORTH: 'Subspecies', LEMMA: 'Subspecies'}]),
                                 ('Virol.', [{ORTH: 'Virology', LEMMA: 'Virology'}]),
                                 ('Tab.', [{ORTH: 'Table', LEMMA: 'Table'}]),
                                 ('Clin.', [{ORTH: 'Clinical', LEMMA: 'clinical'}])]

        if abbreviations is not None:
            default_abbreviations += abbreviations

        for abbreviation in default_abbreviations:
            nlp.tokenizer.add_special_case(*abbreviation)
