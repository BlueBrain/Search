"""Collection of tests regarding the Database creation. """
from bbsearch.sql import find_paragraph, get_paragraph_ids


class TestSQLQueries:

    def test_find_paragraph(self, fake_db_cursor, fake_db_cnxn, fake_slqalchemy_cnxn):
        """Test that the find paragraph method is working."""
        sentence_id = 7
        sentence_text = fake_db_cursor.execute('SELECT text FROM sentences WHERE sentence_id = ?',
                                               [sentence_id]).fetchone()[0]
        paragraph_text = find_paragraph(sentence_id, fake_db_cnxn)
        assert sentence_text in paragraph_text
        paragraph_text = find_paragraph(sentence_id, fake_slqalchemy_cnxn)
        assert sentence_text in paragraph_text

    def test_get_paragraph_ids(self, test_parameters, fake_db_cnxn, fake_slqalchemy_cnxn):
        article_ids = ['w8579f54', '4vo7n6nh', 'agmr7s98', 'i1t9ruf6', 'hnm54k4r', 'tyun4zlh']
        res = get_paragraph_ids(article_ids, fake_db_cnxn)

        assert set(res.unique()) == set(article_ids)
        assert len(res) == len(article_ids) * test_parameters['n_sections_per_article']

        res = get_paragraph_ids(article_ids, fake_slqalchemy_cnxn)

        assert set(res.unique()) == set(article_ids)
        assert len(res) == len(article_ids) * test_parameters['n_sections_per_article']
