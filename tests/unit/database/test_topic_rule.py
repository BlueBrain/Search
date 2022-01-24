# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import re

import pytest

from bluesearch.database.article import ArticleSource
from bluesearch.database.topic_info import TopicInfo
from bluesearch.entrypoint.database.topic_filter import TopicRule


class TestTopicRule:
    def test_noparams(self):
        rule = TopicRule()

        assert rule.level is None
        assert rule.source is None
        assert rule.pattern is None

    def test_level_validation(self):
        rule_1 = TopicRule(level="article")
        rule_2 = TopicRule(level="journal")

        assert rule_1.level == "article"
        assert rule_2.level == "journal"

        with pytest.raises(ValueError, match="Unsupported level"):
            TopicRule(level="wrong")

    def test_source_validation(self):
        rule_1 = TopicRule(source="arxiv")
        rule_2 = TopicRule(source=ArticleSource("biorxiv"))
        rule_3 = TopicRule(source=ArticleSource.PUBMED)

        assert rule_1.source is ArticleSource.ARXIV
        assert rule_2.source is ArticleSource.BIORXIV
        assert rule_3.source is ArticleSource.PUBMED

        with pytest.raises(ValueError, match="Unsupported source"):
            TopicRule(source="wrong_source")

    def test_pattern_validation(self):
        rule_1 = TopicRule(pattern="some_pattern")
        rule_2 = TopicRule(pattern=re.compile("whatever"))

        assert rule_1.pattern.pattern == "some_pattern"
        assert rule_2.pattern.pattern == "whatever"

        with pytest.raises(ValueError, match="Unsupported pattern"):
            TopicRule(pattern=r"\x")

    def test_matching(self):
        info = TopicInfo.from_dict(
            {
                "source": "arxiv",
                "path": "some_path",
                "topics": {
                    "article": {
                        "some_key": ["book", "food"],
                        "some_other_key": ["meat"],
                    },
                    "journal": {
                        "some_key": ["pasta"],
                    },
                },
            }
        )

        rule_1 = TopicRule(pattern="oo")
        rule_2 = TopicRule()
        rule_3 = TopicRule(level="journal", pattern="asta")
        rule_4 = TopicRule(source="biorxiv")
        rule_5 = TopicRule(level="article", pattern="eat")
        rule_6 = TopicRule(level="article", pattern="eataaa")
        rule_7 = TopicRule(level="journal", pattern="837214")

        assert rule_1.match(info)
        assert rule_2.match(info)
        assert rule_3.match(info)
        assert not rule_4.match(info)
        assert rule_5.match(info)
        assert not rule_6.match(info)
        assert not rule_7.match(info)