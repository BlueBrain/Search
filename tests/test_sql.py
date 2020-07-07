"""Collection of tests regarding the Database creation. """
import sqlalchemy
from sqlalchemy.orm import sessionmaker

from bbsearch.sql import find_paragraph, get_paragraph_ids, Sentences


class TestSQLQueries:

    def test_find_paragraph(self, fake_db_cnxn):
        """Test that the find paragraph method is working."""
        database_path = fake_db_cnxn.execute("""PRAGMA database_list""").fetchall()[0][2]
        engine = sqlalchemy.create_engine(f'sqlite:///{database_path}')
        session = sessionmaker(bind=engine)()
        sentence_id = 7
        sentence_text = session.query(Sentences.text).filter(Sentences.sentence_id == sentence_id).one()[0]
        paragraph_text = find_paragraph(sentence_id, session)
        assert sentence_text in paragraph_text

    def test_get_paragraph_ids(self, test_parameters, fake_db_cnxn):
        article_ids = ['w8579f54', '4vo7n6nh', 'agmr7s98', 'i1t9ruf6', 'hnm54k4r', 'tyun4zlh']
        res = get_paragraph_ids(article_ids, fake_db_cnxn)

        assert set(res.unique()) == set(article_ids)
        assert len(res) == len(article_ids) * test_parameters['n_sections_per_article']
