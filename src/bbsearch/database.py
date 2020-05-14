"""Module for the Database Creation."""
import json
import pandas as pd
from pathlib import Path
import re
import sqlite3

from spacy.lang.en import English
from spacy.attrs import ORTH, LEMMA


class DatabaseCreation:
    """Create SQL database from a specified dataset."""

    def __init__(self,
                 data_path,
                 version,
                 saving_directory=None):
        """Creates SQL database object.

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
        if not Path(self.data_path).exists():
            raise NotADirectoryError(f'The data directory {self.data_path} does not exit')

        self.version = version

        self.saving_directory = saving_directory or Path.cwd()
        if not Path(self.saving_directory).exists():
            raise NotADirectoryError(f'The saving directory {self.saving_directory} does not exit')

        self.filename = self.saving_directory / f'cord19_{self.version}.db'

        self.metadata = pd.read_csv(self.data_path / 'metadata.csv')

    def construct(self):
        """Constructs the database."""

        self._rename_columns()
        self._schema_creation()
        self._article_id_to_sha_table()
        self._articles_table()
        self._paragraphs_table()
        self._sentences_table()

    def _schema_creation(self):
        """Creation of the schemas of the different tables in the database. """
        if self.filename.exists():
            raise ValueError(f'The version {self.version} of the database already exists')
        else:
            with sqlite3.connect(str(self.filename)) as db:
                db.execute(
                    """CREATE TABLE IF NOT EXISTS article_id_2_sha
                    (
                        article_id TEXT,
                        sha TEXT
                    );
                    """)
                db.execute(
                    """CREATE TABLE IF NOT EXISTS articles
                    (
                        article_id TEXT PRIMARY KEY,
                        publisher TEXT,
                        title TEXT,
                        doi TEXT,
                        pmc_id TEXT,
                        pm_id INTEGER,
                        licence TEXT,
                        abstract TEXT,
                        date DATETIME,
                        authors TEXT,
                        journal TEXT,
                        microsoft_id INTEGER,
                        covidence_id TEXT,
                        has_pdf_parse BOOLEAN,
                        has_pmc_xml_parse BOOLEAN,
                        has_covid19_tag BOOLEAN DEFAULT False,
                        fulltext_directory TEXT,
                        url TEXT
                    );
                    """)
                db.execute(
                    """CREATE TABLE paragraphs
                    (
                        paragraph_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sha TEXT,
                        section_name TEXT,
                        text TEXT
                    );
                    """)
                db.execute(
                    """CREATE TABLE sentences
                    (
                        sentence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sha TEXT,
                        section_name TEXT,
                        text TEXT,
                        paragraph_id INTEGER,
                        FOREIGN KEY(sha) REFERENCES article_id_2_sha(sha)
                    );
                    """)

    def _rename_columns(self):
        """Renames the columns of the dataframe to follow the SQL database schema. """
        df = self.metadata
        df.rename(columns={
            'cord_uid': 'article_id',
            'sha': 'sha',
            'source_x': 'publisher',
            'title': 'title',
            'doi': 'doi',
            'pmcid': 'pmc_id',
            'pubmed_id': 'pm_id',
            'license': 'licence',
            'abstract': 'abstract',
            'publish_time': 'date',
            'authors': 'authors',
            'journal': 'journal',
            'Microsoft Academic Paper ID': 'microsoft_id',
            'WHO #Covidence': 'covidence_id',
            'has_pdf_parse': 'has_pdf_parse',
            'has_pmc_xml_parse': 'has_pmc_xml_parse',
            'full_text_file': 'fulltext_directory',
            'url': 'url'}, inplace=True)

    def _articles_table(self):
        """Fills the Article Table thanks to 'metadata.csv'.

        Notes
        -----
        The Dataframe self.metadata is modified in this method.
        The article_id_to_sha should be created before calling this method.
        """
        df = self.metadata.copy()
        df = df[df.columns[~df.columns.isin(['sha'])]]
        df.drop_duplicates('article_id', keep='first', inplace=True)
        with sqlite3.connect(str(self.filename)) as db:
            df.to_sql(name='articles', con=db, index=False, if_exists='append')

    def _article_id_to_sha_table(self):
        """Fills the article_id_to_sha table thanks to 'metadata.csv'. '"""
        df = self.metadata[['article_id', 'sha']]
        df = df.set_index(['article_id']).apply(lambda x: x.str.split('; ').explode()).reset_index()
        with sqlite3.connect(str(self.filename)) as db:
            df.to_sql(name='article_id_2_sha', con=db, index=False, if_exists='append')

    def _paragraphs_table(self):
        """Fill the paragraphs table thanks to all the json files."""
        with sqlite3.connect(str(self.filename)) as db:

            cur = db.cursor()
            for (article_id,) in cur.execute('SELECT article_id FROM articles'):
                tag, paragraphs = get_tag_and_paragraph(db, self.data_path, article_id)
                update_covid19_tag(db, article_id, tag)
                insert_into_paragraphs(db, paragraphs)

            db.commit()

    def _sentences_table(self):
        """Fills the sentences table thanks to all the json files. """
        nlp = define_nlp()
        with sqlite3.connect(str(self.filename)) as db:

            cur = db.cursor()
            for (paragraph_id,) in cur.execute('SELECT paragraph_id FROM paragraphs'):
                sentences = get_sentences(db, nlp, paragraph_id)
                insert_into_sentences(db, sentences)

            db.commit()


def add_abbreviations(nlp, abbreviations=None):
    """Add new abbreviations to the default list to avoid wrong scission. (e.g. Dr., Fig., ...).

    Parameters
    ----------
    nlp : spacy.lang.en.English()
        Spacy NLP used for the sentence boundary detection.
    abbreviations: list of tuples
        New abbreviations to add to the default list. Format: (abbreviation, [{ORTH: value, LEMMA: value}])
    """
    default_abbreviations = [('approx.', [{ORTH: 'approximatively', LEMMA: 'approximatively'}]),
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
                             ('Proc.', [{ORTH: 'Procedings', LEMMA: 'procedings'}]),
                             ('Acad.', [{ORTH: 'Academy', LEMMA: 'Academy'}]),
                             ('No.', [{ORTH: 'Number', LEMMA: 'Number'}]),
                             ('Med.', [{ORTH: 'Medecin', LEMMA: 'medecin'}]),
                             ('Rev.', [{ORTH: 'Review', LEMMA: 'review'}]),
                             ('Subsp.', [{ORTH: 'Subspecies', LEMMA: 'Subspecies'}]),
                             ('Virol.', [{ORTH: 'Virology', LEMMA: 'Virology'}]),
                             ('Tab.', [{ORTH: 'Table', LEMMA: 'Table'}]),
                             ('Clin.', [{ORTH: 'Clinical', LEMMA: 'clinical'}])]

    if abbreviations:
        default_abbreviations.extend(abbreviations)

    for abbreviation in default_abbreviations:
        nlp.tokenizer.add_special_case(*abbreviation)


def define_nlp():
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
    add_abbreviations(nlp)

    return nlp


def segment(nlp, paragraph):
    """Segment an paragraph/article into sentences.

    Parameters
    ----------
    nlp: spacy.language.Language
        Spacy pipeline applying sentence segmentantion.
    paragraph: str
        Paragraph/Article in raw text to segment into sentences.

    Returns
    -------
    all_sentences: list
        List of all the sentences extracted from the paragraph.
    """
    all_sentences = (sent.string.strip() for sent in nlp(paragraph).sents)
    return all_sentences


def remove_sentences_duplicates(sentences):
    """Returns a filtered list of sentences.

    Notes
    ------
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
    keys = set()

    # Boilerplate text to ignore
    boilerplate = ["COVID-19 resource centre",
                   "permission to make all its COVID",
                   "WHO COVID database"]

    for sha, name, text in sentences:
        # Add unique text that isn't boilerplate text
        if text not in keys and not any(x in text for x in boilerplate):
            unique.append((sha, name, text))
            keys.add(text)

    return unique


def get_tags(sentences):
    """Computes the tag for an article id through its sentences.

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
    regex = "|".join(["\\b%s\\b" % keyword.lower() for keyword in keywords])
    tag = False  # None
    for _, _, text in sentences:
        # Look for at least one keyword match
        if re.findall(regex, text.lower()):
            tag = True  # "COVID-19"
    return tag


def get_tag_and_paragraph(db, data_directory, article_id):
    """Extract all the paragraph and the tag has_covid19 from an article.

    Parameters
    ----------
    db:
        Database
    data_directory: Path
        Directory where all the json files are located
    article_id: str
        ID of the article specified in the articles database.

    Returns
    -------
    tag: boolean
        Tag value of has_covid19. This is checking if covid19 is mentionned in the paper.
    paragraphs: list
        List of the extracted paragraphs. (paragraph_id, paragraph)
    """
    paragraphs = []
    tag = False

    article_id, article_title, article_abstract, article_directory = db.execute(
        "SELECT article_id, title, abstract, fulltext_directory FROM articles WHERE article_id is ?",
        [article_id]).fetchone()

    all_shas = db.execute("SELECT sha FROM article_id_2_sha WHERE article_id = ?", [article_id]).fetchall()
    title_sha = all_shas[0][0] if all_shas else None
    if article_title:
        paragraphs.append((title_sha, 'Title', article_title))
    if article_abstract:
        paragraphs.append((title_sha, 'Abstract', article_abstract))

    for (sha,) in all_shas:
        if sha:
            found_json_files = list(data_directory.glob(f'**/*{sha}*json'))
            if len(found_json_files) != 1:
                raise ValueError(f'Found {len(found_json_files)} json files for sha {sha}')
            with open(str(found_json_files[0])) as json_file:
                file = json.load(json_file)
                for sec in file['body_text']:
                    paragraphs.append((sha, sec['section'].title(), sec['text']))
                for _, v in file['ref_entries'].items():
                    paragraphs.append((sha, 'Caption', v['text']))

    tag = tag or get_tags(paragraphs)

    return tag, paragraphs


def get_sentences(db, nlp, paragraph_id):
    """Extract all the sentences from the paragraph table.

    Parameters
    ----------
    db:
        Database
    nlp: spacy.language.Language
        Sentence Boundary Detection tool from Spacy to seperate sentences.
    paragraph_id: int
        ID of the paragraph

    Returns
    -------
    sentences: list
        List of the extracted sentences.
    """
    sentences = []
    sha, section_name, paragraph = db.execute("SELECT sha, section_name, text FROM paragraphs WHERE paragraph_id = ?",
                                              [paragraph_id]).fetchall()[0]
    sentences.extend([(sha, section_name, sent, paragraph_id) for sent in segment(nlp, paragraph)])

    return sentences


def update_covid19_tag(db, article_id, tag):
    """Update the covid19 tag in the articles database.

    Parameters
    ----------
    db: sql database
        Database with the table articles to update.
    article_id: str
        Article ID of the row to update into the database.
    tag: boolean
        Value of the tag. True if covid19 is mentionned, otherwise False.
    """

    db.execute("UPDATE articles SET has_covid19_tag = ? WHERE article_id = ?", [tag, article_id])


def insert_into_sentences(db, sentences):
    """Insert the new sentences into the database sentences.

    Parameters
    ----------
    db: sql database
        Database with the table sentences where to insert new sentences.
    sentences: list
        List of sentences to insert in format (sha, section_name, text, paragraph_id)
    """
    cur = db.cursor()
    cur.executemany("INSERT INTO sentences (sha, section_name, text, paragraph_id) VALUES (?, ?, ?, ?)", sentences)


def insert_into_paragraphs(db, paragraphs):
    """Insert the new sentences into the database sentences.

    Parameters
    ----------
    db: sql database
        Database with the table sentences where to insert new sentences.
    paragraphs: list
        List of sentences to insert in format (paragraph_id, text)
    """
    cur = db.cursor()
    cur.executemany("INSERT INTO paragraphs (sha, section_name, text) VALUES (?, ?, ?)", paragraphs)
