"""Configuration of pytest."""
from pathlib import Path
import time

import docker
import h5py
import numpy as np
import pandas as pd
import pytest
import spacy
import sqlalchemy

ROOT_PATH = Path(__file__).resolve().parent.parent  # root of the repository


@pytest.fixture(scope='session')
def test_parameters():
    """Parameters needed for the tests"""
    return {'n_sentences_per_section': 3,
            'n_sections_per_article': 2,
            'embedding_size': 2}


def fill_db_data(connection, metadata_path, test_parameters):
    articles_schema = {'article_id': 'INTEGER PRIMARY KEY',
                       'cord_uid': 'VARCHAR(8)',
                       'sha': 'TEXT',
                       'source_x': 'TEXT',
                       'title': 'TEXT',
                       'doi': 'TEXT',
                       'pmcid': 'TEXT',
                       'pubmed_id': 'TEXT',
                       'license': 'TEXT',
                       'abstract': 'TEXT',
                       'publish_time': 'DATE',
                       'authors': 'TEXT',
                       'journal': 'TEXT',
                       'mag_id': 'TEXT',
                       'who_covidence_id': 'TEXT',
                       'arxiv_id': 'TEXT',
                       'pdf_json_files': 'TEXT',
                       'pmc_json_files': 'TEXT',
                       'url': 'TEXT',
                       's2_id': 'TEXT'}

    sentences_schema = {'sentence_id': 'INTEGER PRIMARY KEY',
                        'section_name': 'TEXT',
                        'article_id': 'INTEGER',
                        'text': 'TEXT',
                        'paragraph_pos_in_article': 'INTEGER',
                        'sentence_pos_in_paragraph': 'INTEGER',
                        'FOREIGN': 'KEY(article_id) REFERENCES articles(article_id)'}

    stmt_create_articles = "CREATE TABLE articles ({})".format(
        ', '.join(['{} {}'.format(k, v) for k, v in articles_schema.items()]))

    stmt_create_sentences = "CREATE TABLE sentences ({})".format(
        ', '.join(['{} {}'.format(k, v) for k, v in sentences_schema.items()]))

    connection.execute(stmt_create_articles)
    connection.execute(stmt_create_sentences)

    metadata_df = pd.read_csv(str(metadata_path))
    metadata_df['article_id'] = metadata_df.index
    metadata_df.to_sql(name='articles', con=connection, index=False, if_exists='append')

    temp_s = []
    for article_id in set(metadata_df[metadata_df['article_id'].notna()]['article_id'].to_list()):
        for sec_ix in range(test_parameters['n_sections_per_article']):
            for sen_ix in range(test_parameters['n_sentences_per_section']):
                s = pd.Series({'text': 'I am a sentence {} in section {} '
                                       'in article {}.'.format(sen_ix, sec_ix, article_id),
                               'section_name': 'section_{}'.format(sec_ix),
                               'article_id': article_id,
                               'paragraph_pos_in_article': sec_ix,
                               'sentence_pos_in_paragraph': sen_ix,
                               })
                temp_s.append(s)

    sentences_content = pd.DataFrame(temp_s)
    sentences_content.index.name = 'sentence_id'
    sentences_content.index += 1
    sentences_content.to_sql(name='sentences', con=connection, index=True, if_exists='append')


@pytest.fixture(scope='session', params=['sqlite', 'mysql'])
def fake_sqlalchemy_engine(tmp_path_factory, metadata_path, test_parameters, request):
    """Connection object (sqlite)."""
    if request.param == 'sqlite':
        db_path = tmp_path_factory.mktemp('db', numbered=False) / 'cord19_test.db'
        Path(db_path).touch()
        engine = sqlalchemy.create_engine(f'sqlite:///{db_path}')
        fill_db_data(engine, metadata_path, test_parameters)
        yield engine

    else:
        client = docker.from_env()
        container = client.containers.run('mysql:latest',
                                          environment={'MYSQL_ROOT_PASSWORD': 'my-secret-pw'},
                                          ports={'3306/tcp': 3306},
                                          detach=True)

        docker_ready = False

        max_waiting_time = 2 * 60
        start = time.perf_counter()

        while not docker_ready and (time.perf_counter() - start) < max_waiting_time:
            try:
                engine = sqlalchemy.create_engine('mysql+pymysql://root:my-secret-pw'
                                                  '@127.0.0.1:3306/')
                engine.execute('show databases')
            except sqlalchemy.exc.OperationalError:
                time.sleep(5)

                continue
            docker_ready = True

        engine.execute("create database test")
        engine.execute("use test")
        fill_db_data(engine, metadata_path, test_parameters)

        yield engine

        container.kill()


@pytest.fixture(scope='session')
def fake_sqlalchemy_cnxn(fake_sqlalchemy_engine):
    """Connection object (sqlite)."""
    sqlalchemy_cnxn = fake_sqlalchemy_engine.connect()
    return sqlalchemy_cnxn


@pytest.fixture(scope='session')
def jsons_path():
    """Path to a directory where jsons are stored."""
    jsons_path = ROOT_PATH / 'tests' / 'data' / 'cord19_v35'
    assert jsons_path.exists()

    return jsons_path


@pytest.fixture(scope='session')
def metadata_path():
    """Path to metadata.csv."""
    metadata_path = ROOT_PATH / 'tests' / 'data' / 'cord19_v35' / 'metadata.csv'
    assert metadata_path.exists()

    return metadata_path


@pytest.fixture(scope='session')
def embeddings_h5_path(tmp_path_factory, fake_sqlalchemy_engine, test_parameters):
    random_state = 3
    np.random.seed(random_state)
    models = ['SBERT', 'SBioBERT', 'USE', 'BSV']
    dim = test_parameters['embedding_size']
    n_sentences = pd.read_sql('SELECT COUNT(*) FROM sentences', fake_sqlalchemy_engine).iloc[0, 0]
    file_path = tmp_path_factory.mktemp('h5_embeddings', numbered=False) / 'embeddings.h5'

    with h5py.File(file_path) as f:
        for model in models:
            dset = f.create_dataset(f"{model}",
                                    (n_sentences, dim),
                                    dtype='f4',
                                    fillvalue=np.nan)

            for i in range(0, n_sentences, 2):
                dset[i] = np.random.random(dim).astype('float32')

    return file_path


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


@pytest.fixture(scope='session')
def punctuation_annotations():
    files_location = ROOT_PATH / 'tests' / 'data' / 'mining' / 'eval'
    return {mode: pd.read_csv(files_location / f'iob_punctuation_{mode}.csv')
            for mode in ('before', 'after')}
