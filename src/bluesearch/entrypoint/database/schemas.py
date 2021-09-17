"""Module for defining SQL schemas."""

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
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

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)


def schema_articles(metadata: MetaData) -> None:
    """Add to the given 'metadata' the schema of the table 'articles'."""
    Table(
        "articles",
        metadata,
        Column("article_id", String(32), primary_key=True),
        Column("doi", Text()),
        Column("pmc_id", Text()),
        Column("pubmed_id", Text()),
        Column("title", Text()),
        Column("authors", Text()),
        Column("abstract", Text()),
        Column("journal", Text()),
        Column("publish_time", Date()),
        Column("license", Text()),
        Column("is_english", Boolean()),
    )


def schema_sentences(metadata: MetaData) -> None:
    """Add to the given 'metadata' the schema of the table 'sentences'."""
    Table(
        "sentences",
        metadata,
        Column("sentence_id", Integer(), primary_key=True, autoincrement=True),
        Column("section_name", Text()),
        Column("text", Text()),
        Column(
            "article_id", String(32), ForeignKey("articles.article_id"), nullable=False
        ),
        Column("paragraph_pos_in_article", Integer(), nullable=False),
        Column("sentence_pos_in_paragraph", Integer(), nullable=False),
        UniqueConstraint(
            "article_id",
            "paragraph_pos_in_article",
            "sentence_pos_in_paragraph",
            name="sentence_unique_identifier",
        ),
        Column("is_bad", Boolean(), server_default="0"),
    )
