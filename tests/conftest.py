"""Configuration of pytest."""
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

ROOT_PATH = Path(__file__).resolve().parent.parent  # root of the repository
N_SENTENCES_PER_SECTION = 3
N_SECTIONS_PER_ARTICLE = 2
EMBEDDING_SIZE = 2


@pytest.fixture(scope='session')
def assets_path(tmp_path_factory):
    """Path to assets."""
    assets_path = tmp_path_factory.mktemp('assets', numbered=False)

    return assets_path


@pytest.fixture(scope='session')
def cnxn(tmp_path_factory):
    """Connection object (sqlite)."""
    db_path = tmp_path_factory.mktemp('db', numbered=False) / 'dummy.sqlite'
    cnxn = sqlite3.connect(str(db_path))
    yield cnxn

    cnxn.close()  # disconnect


@pytest.fixture(scope='session')
def cursor(cnxn, jsons_path, metadata_path):
    """Database object (sqlite)."""
    cursor = cnxn.cursor()
    # create

    articles_schema = {'article_id': 'TEXT PRIMARY KEY',
                       'publisher': 'TEXT',
                       'title': 'TEXT',
                       'doi': 'TEXT',
                       'pmc_id': 'TEXT',
                       'pm_id': 'INTEGER',
                       'licence': 'TEXT',
                       'abstract': 'TEXT',
                       'date': 'DATETIME',
                       'authors': 'TEXT',
                       'journal': 'TEXT',
                       'microsoft_id': 'INTEGER',
                       'covidence_id': 'TEXT',
                       'has_pdf_parse': 'BOOLEAN',
                       'has_pmc_xml_parse': 'BOOLEAN',
                       'has_covid19_tag': 'BOOLEAN DEFAULT False',
                       'fulltext_directory': 'TEXT',
                       'url': 'TEXT'}

    article_id_2_sha_schema = {'article_id': 'TEXT',
                               'sha': 'TEXT'}

    sentences_schema = {'sentence_id': 'INTEGER PRIMARY KEY',
                        'sha': 'TEXT',
                        'section_name': 'TEXT',
                        'text': 'TEXT',
                        'FOREIGN': 'KEY(sha) REFERENCES article_id_2_sha(sha)'}

    stmt_create_articles = """CREATE TABLE articles ({})""".format(
        ', '.join(['{} {}'.format(k, v) for k, v in articles_schema.items()]))

    stmt_create_id_2_sha = """CREATE TABLE article_id_2_sha ({})""".format(
        ', '.join(['{} {}'.format(k, v) for k, v in article_id_2_sha_schema.items()]))

    stmt_create_sentences = """CREATE TABLE sentences ({})""".format(
        ', '.join(['{} {}'.format(k, v) for k, v in sentences_schema.items()]))

    cursor.execute(stmt_create_articles)
    cursor.execute(stmt_create_id_2_sha)
    cursor.execute(stmt_create_sentences)

    # Populate
    name_mapping = {
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
        'url': 'url'
    }

    metadata_df = pd.read_csv(str(metadata_path)).rename(columns=name_mapping).set_index('article_id')

    article_id_2_content = metadata_df['sha']
    article_id_2_content.to_sql(name='article_id_2_sha', con=cnxn, index=True, if_exists='append')

    articles_content = metadata_df.drop(columns=['sha'])
    articles_content.to_sql(name='articles', con=cnxn, index=True, if_exists='append')

    temp = []
    for sha in article_id_2_content[article_id_2_content.notna()].unique():
        for sec_ix in range(N_SECTIONS_PER_ARTICLE):
            for sen_ix in range(N_SENTENCES_PER_SECTION):
                s = pd.Series({'text': 'I am a sentence {} in section {} in article {}'.format(sen_ix, sec_ix, sha),
                               'section_name': 'section_{}'.format(sec_ix),
                               'sha': sha
                               })
                temp.append(s)

    sentences_content = pd.DataFrame(temp)
    sentences_content.index.name = 'sentence_id'
    sentences_content.to_sql(name='sentences', con=cnxn, index=True, if_exists='append')

    yield cursor

    cnxn.rollback()  # undo uncommited changes -> after tests are run all changes are deleted INVESTIGATE


@pytest.fixture(scope='session')
def jsons_path():
    """Path to a directory where jsons are stored."""
    jsons_path = ROOT_PATH / 'tests' / 'data' / 'CORD19_samples'
    assert jsons_path.exists()

    return jsons_path


@pytest.fixture(scope='session')
def metadata_path():
    """Path to metadata.csv."""
    metadata_path = ROOT_PATH / 'tests' / 'data' / 'CORD19_samples' / 'metadata.csv'
    assert metadata_path.exists()

    return metadata_path


@pytest.fixture(scope='session')
def embeddings_path(tmp_path_factory, cursor):
    """Path to a directory where embeddings stored."""
    random_state = 3
    np.random.seed(random_state)
    models = ['BERT', 'BIOBERT']

    n_sentences = cursor.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]
    embeddings_path = tmp_path_factory.mktemp('embeddings', numbered=False)

    for model in models:
        model_path = embeddings_path / model / '{}.npy'.format(model)
        model_path.parent.mkdir(parents=True)
        a = np.concatenate([np.arange(n_sentences).reshape(-1, 1), np.random.random((n_sentences, EMBEDDING_SIZE))],
                           axis=1)

        np.save(str(model_path), a)

    return embeddings_path
