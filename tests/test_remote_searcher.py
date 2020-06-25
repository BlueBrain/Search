from unittest.mock import Mock

import pytest
from requests.models import Response

from bbsearch.remote_searcher import RemoteSearcher


@pytest.mark.parametrize('status', [True, False])
def test_all(monkeypatch, status):
    fake_response = Mock(spec=Response)
    fake_response.json.return_value = {'sentence_ids': 'A',
                                       'similarities': 'B',
                                       'stats': 'C'}

    fake_response.ok = status

    fake_requests = Mock()
    fake_requests.post.return_value = fake_response

    monkeypatch.setattr('bbsearch.remote_searcher.requests', fake_requests)
    searcher = RemoteSearcher('')
    sentence_ids, similarities, stats = searcher.query('model', 2, 'some_text')

    if status:
        assert sentence_ids == 'A'
        assert similarities == 'B'
        assert stats == 'C'
    else:
        assert sentence_ids is None
        assert similarities is None
        assert stats is None

    assert fake_response.json.call_count == int(status)
    assert fake_requests.post.call_count == 1
