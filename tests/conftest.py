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


def fill_db_data(engine, metadata_path, test_parameters):
    metadata = sqlalchemy.MetaData()

    articles_table = \
        sqlalchemy.Table('articles', metadata,
                         sqlalchemy.Column('article_id', sqlalchemy.Integer(),
                                           primary_key=True, autoincrement=True),
                         sqlalchemy.Column('cord_uid', sqlalchemy.String(8), nullable=False),
                         sqlalchemy.Column('sha', sqlalchemy.Text()),
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

    sentences_table = \
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

    with engine.begin() as connection:
        metadata.create_all(connection)

    mymodel_url_index = sqlalchemy.Index('article_id_index', sentences_table.c.article_id)
    mymodel_url_index.create(bind=engine)

    metadata_df = pd.read_csv(str(metadata_path))
    metadata_df.index.name = 'article_id'
    metadata_df.index += 1
    metadata_df.to_sql(name='articles', con=engine, index=True, if_exists='append')

    temp_s = []
    for article_id in set(metadata_df[metadata_df.index.notna()].index.to_list()):
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
    sentences_content.to_sql(name='sentences', con=engine, index=True, if_exists='append')


@pytest.fixture(scope='session', params=['sqlite', 'mysql'])
def backend_database(request):

    return request.param


@pytest.fixture(scope='session')
def fake_sqlalchemy_engine(tmp_path_factory, metadata_path, test_parameters, backend_database):
    """Connection object (sqlite)."""
    if backend_database == 'sqlite':
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
        engine.dispose()
        engine = sqlalchemy.create_engine('mysql+pymysql://root:my-secret-pw'
                                          '@127.0.0.1:3306/test')
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
    if not (tmp_path_factory.getbasetemp() / 'h5_embeddings').is_dir():
        file_path = tmp_path_factory.mktemp('h5_embeddings', numbered=False) / 'embeddings.h5'

        with h5py.File(file_path) as f:
            for model in models:
                dset = f.create_dataset(f"{model}",
                                        (n_sentences, dim),
                                        dtype='f4',
                                        fillvalue=np.nan)

                for i in range(0, n_sentences, 2):
                    dset[i] = np.random.random(dim).astype('float32')
    else:
        file_path = tmp_path_factory.getbasetemp() / 'h5_embeddings' / 'embeddings.h5'

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
