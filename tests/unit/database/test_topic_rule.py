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
from __future__ import annotations

import re

import pytest

from bluesearch.database.article import ArticleSource
from bluesearch.database.topic_info import TopicInfo
from bluesearch.database.topic_rule import TopicRule, check_accepted


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

        assert rule_1.pattern is not None
        assert rule_2.pattern is not None
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
                "metadata": {},
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


def test_check_accepted():
    topic_info = TopicInfo.from_dict(
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
            "metadata": {},
        }
    )

    # No rules specified
    topic_rules_accept: list[TopicRule] = []
    topic_rules_reject: list[TopicRule] = []
    assert not check_accepted(topic_info, topic_rules_accept, topic_rules_reject)

    # Nothing matches
    topic_rules_accept = [
        TopicRule(source="biorxiv"),
    ]
    topic_rules_reject = [
        TopicRule(level="journal", pattern="837214"),
        TopicRule(level="article", pattern="eataaa"),
    ]
    assert not check_accepted(topic_info, topic_rules_accept, topic_rules_reject)

    # One reject rule is matching (no matter if one accept rule is matching)
    topic_rules_accept = [
        TopicRule(source="biorxiv"),
        TopicRule(level="journal", pattern="asta"),
    ]
    topic_rules_reject = [
        TopicRule(source="pmc"),
        TopicRule(source="arxiv", level="article", pattern="oo"),
    ]
    assert not check_accepted(topic_info, topic_rules_accept, topic_rules_reject)

    # No reject rule is matching but one accept rule is matching
    topic_rules_accept = [
        TopicRule(source="biorxiv"),
        TopicRule(level="journal", pattern="asta"),
    ]
    topic_rules_reject = [
        TopicRule(source="pmc"),
    ]
    assert check_accepted(topic_info, topic_rules_accept, topic_rules_reject)
