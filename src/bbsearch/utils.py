"""

Generic Utils waiting for migration in proper submodule.

"""
import json
import re
import time

from spacy.lang.en import English
from spacy.attrs import ORTH, LEMMA


class Timer:
    r"""Convenience context manager timing functions and logging the results.

    The order of execution is `__call__`,  `__enter__` and `__exit__`.

    Parameters
    ----------
    verbose : bool
        If True, whenever process ends we print the elapsed time to standard output.

    Attributes
    ----------
    inst_time : float
        Time of instantiation.

    name : str or None
        Name of the process to be timed. The user can control the value via the `__call__` magic.

    logs : dict
        Internal dictionary that stores all the times. The keys are the process names and the values are number
        of seconds.

    start_time : float or None
        Time of the last enter. Is dynamically changed when entering.

    Examples
    --------
    >>> import time
    >>> from bbsearch.utils import Timer
    >>>
    >>> timer = Timer(verbose=False)
    >>>
    >>> with timer('experiment_1'):
    ...     time.sleep(0.05)
    >>>
    >>> with timer('experiment_2'):
    ...     time.sleep(0.02)
    >>>
    >>> assert set(timer.stats.keys()) == {'overall', 'experiment_1', 'experiment_2'}
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

        self.inst_time = time.time()
        self.name = None  # what key is being populated
        self.logs = {}
        self.start_time = None  # to be overwritten when entering

    def __call__(self, name, message=None):
        """Define the name of the process to be timed.

        Parameters
        ----------
        name : str
            Name of the process to be timed.

        message : str or None
            Optional message to be printed to stoud when entering. Note that it only has an effect if
            `self.verbose=True`.

        """
        self.name = name

        if self.verbose and message is not None:
            print(message)

        return self

    def __enter__(self):
        """Launch the timer."""
        if self.name is None:
            raise ValueError('No name specified, one needs to call the instance with some name.')

        if self.name in self.logs:
            raise ValueError('{} has already been timed'.format(self.name))

        if self.name == 'overall':
            raise ValueError("The 'overall' key is restricted for length of the lifetime of the Timer.")

        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and log internally."""
        if exc_type is not None:
            # raised an exception
            self.start_time = None
            self.name = None
            return False

        else:
            # nothing bad happened
            end_time = time.time()
            self.logs[self.name] = end_time - self.start_time

            if self.verbose:
                fmt = '{:.2f}'
                print("{} took ".format(self.name) + fmt.format(self.logs[self.name]) + ' seconds')

        # cleanup
        self.start_time = None
        self.name = None

    def __getitem__(self, item):
        """Get a single experiment."""
        return self.logs[item]

    @property
    def stats(self):
        """Return all timing statistics."""
        return {'overall': time.time() - self.inst_time, **self.logs}


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


def segment(nlp, sentences):
    """Segment an paragraph/article into sentences.

    Parameters
    ----------
    nlp: spacy.language.Language
        Spacy pipeline applying sentence segmentantion.
    sentences: str
        Paragraph/Article in raw text to segment into sentences.

    Returns
    -------
    all_sentences: list
        List of all the sentences extracted from the paragraph.
    """
    all_sentences = (sent.string.strip() for sent in nlp(sentences).sents)
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


def get_tag_and_sentences(db, nlp, data_directory, article_id):
    """Extract all the sentences and the tag has_covid19 from an article.

    Parameters
    ----------
    db:
        Database
    nlp: spacy.language.Language
        Sentence Boundary Detection tool from Spacy to seperate sentences.
    data_directory: Path
        Directory where all the json files are located
    article_id: str
        ID of the article specified in the articles database.

    Returns
    -------
    tag: boolean
        Tag value of has_covid19. This is checking if covid19 is mentionned in the paper.
    sentences: list
        List of the extracted sentences
    """
    sentences = []
    tag = False

    article_id, article_title, article_abstract, article_directory = db.execute(
        "SELECT article_id, title, abstract, fulltext_directory FROM articles WHERE article_id is ?",
        [article_id]).fetchone()

    all_shas = db.execute("SELECT sha FROM article_id_2_sha WHERE article_id = ?", [article_id]).fetchall()
    title_sha = all_shas[0][0] if all_shas else None
    if article_title:
        sentences.extend([(title_sha, 'Title', sent) for sent in segment(nlp, article_title)])
    if article_abstract:
        sentences.extend([(title_sha, 'Abstract', sent) for sent in segment(nlp, article_abstract)])

    for (sha,) in all_shas:
        if sha:
            found_json_files = list(data_directory.glob(f'**/*{sha}*json'))
            if len(found_json_files) != 1:
                raise ValueError(f'Found {len(found_json_files)} json files for sha {sha}')
            with open(str(found_json_files[0])) as json_file:
                file = json.load(json_file)
                for sec in file['body_text']:
                    sentences.extend([(sha, sec['section'].title(), sent) for sent in segment(nlp, sec['text'])])
                for _, v in file['ref_entries'].items():
                    sentences.extend([(sha, 'Caption', sent) for sent in segment(nlp, v['text'])])

    sentences = remove_sentences_duplicates(sentences)
    tag = tag or get_tags(sentences)

    return tag, sentences


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
        List of sentences to insert in format (sha, section_name, text)
    """
    cur = db.cursor()
    cur.executemany("INSERT INTO sentences (sha, section_name, text) VALUES (?, ?, ?)", sentences)
