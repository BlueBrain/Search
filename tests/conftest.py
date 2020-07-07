"""Configuration of pytest."""
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest
import spacy

ROOT_PATH = Path(__file__).resolve().parent.parent  # root of the repository


@pytest.fixture(scope='session')
def test_parameters():
    """Parameters needed for the tests"""
    return {'n_sentences_per_section': 3,
            'n_sections_per_article': 2,
            'embedding_size': 2}


@pytest.fixture(scope='session')
def fake_db_cnxn(tmp_path_factory):
    """Connection object (sqlite)."""
    db_path = tmp_path_factory.mktemp('db', numbered=False) / 'cord19.db'
    cnxn = sqlite3.connect(str(db_path))
    yield cnxn

    cnxn.close()  # disconnect


@pytest.fixture(scope='session')
def fake_db_cursor(fake_db_cnxn, jsons_path, metadata_path, test_parameters):
    """Database object (sqlite)."""
    cursor = fake_db_cnxn.cursor()
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
                       'has_covid19_tag': 'BOOLEAN DEFAULT 1',
                       'fulltext_directory': 'TEXT',
                       'url': 'TEXT'}

    article_id_2_sha_schema = {'article_id': 'TEXT',
                               'sha': 'TEXT'}

    paragraphs_schema = {'paragraph_id': 'INTEGER PRIMARY KEY',
                         'sha': 'TEXT',
                         'section_name': 'TEXT',
                         'text': 'TEXT',
                         'FOREIGN': 'KEY(sha) REFERENCES article_id_2_sha(sha)'}

    sentences_schema = {'sentence_id': 'INTEGER PRIMARY KEY',
                        'sha': 'TEXT',
                        'section_name': 'TEXT',
                        'text': 'TEXT',
                        'paragraph_id': 'INTEGER',
                        'FOREIGN': 'KEY(sha) REFERENCES article_id_2_sha(sha)'}

    stmt_create_articles = "CREATE TABLE articles ({})".format(
        ', '.join(['{} {}'.format(k, v) for k, v in articles_schema.items()]))

    stmt_create_id_2_sha = "CREATE TABLE article_id_2_sha ({})".format(
        ', '.join(['{} {}'.format(k, v) for k, v in article_id_2_sha_schema.items()]))

    stmt_create_paragraphs = "CREATE TABLE paragraphs ({})".format(
        ', '.join(['{} {}'.format(k, v) for k, v in paragraphs_schema.items()]))

    stmt_create_sentences = "CREATE TABLE sentences ({})".format(
        ', '.join(['{} {}'.format(k, v) for k, v in sentences_schema.items()]))

    cursor.execute(stmt_create_articles)
    cursor.execute(stmt_create_id_2_sha)
    cursor.execute(stmt_create_paragraphs)
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
    article_id_2_content.to_sql(name='article_id_2_sha', con=fake_db_cnxn, index=True, if_exists='append')

    articles_content = metadata_df.drop(columns=['sha'])
    articles_content.to_sql(name='articles', con=fake_db_cnxn, index=True, if_exists='append')

    temp_s = []
    temp_p = []
    paragraph_id = 0
    for sha in article_id_2_content[article_id_2_content.notna()].unique():
        for sec_ix in range(test_parameters['n_sections_per_article']):
            paragraph_text = ''
            for sen_ix in range(test_parameters['n_sentences_per_section']):
                s = pd.Series({'text': 'I am a sentence {} in section {} in article {}.'.format(sen_ix, sec_ix, sha),
                               'section_name': 'section_{}'.format(sec_ix),
                               'sha': sha,
                               'paragraph_id': paragraph_id
                               })
                temp_s.append(s)
                paragraph_text += s['text']

            p = pd.Series({'text': paragraph_text,
                           'section_name': 'section_{}'.format(sec_ix),
                           'sha': sha})
            temp_p.append(p)
            paragraph_id += 1

    sentences_content = pd.DataFrame(temp_s)
    sentences_content.index.name = 'sentence_id'
    sentences_content.to_sql(name='sentences', con=fake_db_cnxn, index=True, if_exists='append')

    paragraphs_content = pd.DataFrame(temp_p)
    paragraphs_content.index.name = 'paragraph_id'
    paragraphs_content.to_sql(name='paragraphs', con=fake_db_cnxn, index=True, if_exists='append')

    yield cursor

    fake_db_cnxn.rollback()  # undo uncommited changes -> after tests are run all changes are deleted INVESTIGATE


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
def embeddings_path(tmp_path_factory, fake_db_cursor, test_parameters):
    """Path to a directory where embeddings stored."""
    random_state = 3
    np.random.seed(random_state)
    models = ['SBERT', 'SBioBERT', 'USE', 'BSV']

    n_sentences = fake_db_cursor.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]
    embeddings_path = tmp_path_factory.mktemp('embeddings', numbered=False)

    for model in models:
        model_path = embeddings_path / '{}.npy'.format(model)
        a = np.concatenate([np.arange(n_sentences).reshape(-1, 1),
                            np.random.random((n_sentences, test_parameters['embedding_size']))],
                           axis=1)

        np.save(str(model_path), a)

    return embeddings_path


@pytest.fixture(scope='session')
def model_entities():
    """Standard English spacy model.

    References
    ----------
    https://spacy.io/api/annotation#named-entities
    """
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope='session')
def ner_annotations():
    csv_filename = ROOT_PATH / 'tests' / 'data' / 'mining' / 'eval' / 'ner_iob_sample.csv'

    return {
        'bio':
            pd.read_csv(csv_filename),
        'sample':
            pd.DataFrame(data={
                'annotator_1': ['B-a', 'B-a', 'B-b', 'B-a', 'O', 'B-a', 'I-a', 'O', 'B-b', 'I-b',
                                'O', 'O', 'B-d', 'B-b'],
                'annotator_2': ['B-c', 'B-c', 'I-c', 'B-c', 'O', 'B-c', 'O', 'B-b', 'I-b', 'I-b',
                                'B-c', 'I-c', 'B-c', 'B-b']
            })
    }
