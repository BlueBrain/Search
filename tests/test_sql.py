"""Collection of tests regarding the Database creation. """
from bbsearch.sql import find_paragraph


class TestSQLQueries:

    def test_find_paragraph(self, fake_db_cursor):
        """Test that the find paragraph method is working."""
        sentence_id = 7
        sentence_text = fake_db_cursor.execute('SELECT text FROM sentences WHERE sentence_id = ?',
                                               [sentence_id]).fetchone()[0]
        paragraph_text = find_paragraph(sentence_id, fake_db_cursor)
        assert sentence_text in paragraph_text
