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
"""Run the overall pipeline."""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
import tarfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterator

import luigi
from luigi.util import inherits, requires
from luigi.contrib.external_program import ExternalProgramTask

from bluesearch.database.article import ArticleSource

logger = logging.getLogger(__name__)


def init_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initialise the argument parser for the run subcommand.

    Parameters
    ----------
    parser
        The argument parser to initialise.

    Returns
    -------
    argparse.ArgumentParser
        The initialised argument parser. The same object as the `parser`
        argument.
    """
    parser.description = "Run the overall pipeline."

    parser.add_argument(
        "source",
        type=str,
        choices=[member.value for member in ArticleSource],
        help="Source of the articles.",
    )
    parser.add_argument(
        "from_month",
        type=str,
        help="The starting month (included) for the download in format YYYY-MM. "
        "All papers from the given month until today will be downloaded.",
    )
    parser.add_argument(
        "filter_config",
        type=Path,
        help="""
        Path to a .JSONL file that defines all the rules for filtering.
        """,
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="""
        Path to the output folder. All the results stored under
        `output_dir/source/date` where date is concatenation of the
        `from_month` and the day of execution of this command.
        """,
    )
    parser.add_argument(
        "db_url",
        type=str,
        help="""
        The location of the database depending on the database type.

        For MySQL and MariaDB the server URL should be provided, for SQLite the
        location of the database file. Generally, the scheme part of
        the URL should be omitted, e.g. for MySQL the URL should be
        of the form 'my_sql_server.ch:1234/my_database' and for SQLite
        of the form '/path/to/the/local/database.db'.
        """,
    )
    parser.add_argument(
        "--db-type",
        default="sqlite",
        type=str,
        choices=("mariadb", "mysql", "postgres", "sqlite"),
        help="Type of the database.",
    )
    parser.add_argument(
        "--mesh-topic-db",
        type=Path,
        help="""
        The JSON file with MeSH topic hierarchy information. Mandatory for
        source types "pmc" and "pubmed".

        The JSON file should contain a flat dictionary with MeSH topic tree
        numbers mapped to the corresponding topic labels. This file can be
        produced using the `bbs_database parse-mesh-rdf` command. See that
        command's description for more details.
        """,
    )
    return parser


BBS_BINARY = "bbs_database"
CAPTURE_OUTPUT = False

class DownloadTask(ExternalProgramTask):
    source = luigi.Parameter()
    from_month = luigi.Parameter()
    output_dir = luigi.Parameter()


    def output(self):
        today = datetime.today()
        date = f"{self.from_month}_{today.strftime('%Y-%m-%d')}"

        output_dir = Path(self.output_dir) / self.source / date / "raw"

        return luigi.LocalTarget(str(output_dir))


    def program_args(self):
        output_dir = self.output().path
        return [
            BBS_BINARY, "download", "-v", self.source, self.from_month, output_dir,
        ]

@requires(DownloadTask)
class UnzipTask(ExternalProgramTask):
    """Needs to support unziping of both pubmed and pmc."""
    source = luigi.Parameter()


    def output(self):
        input_path = Path(self.input().path)
        output_dir = input_path.parent / "raw_unzipped"

        return luigi.LocalTarget(str(output_dir))

    def run(self):
        input_dir =  Path(self.input().path) # raw
        output_dir = Path(self.output().path)  # raw_unzipped

        
        output_dir.mkdir(exist_ok=True, parents=True)
        if self.source == "pmc":
            # .tar.gz
            # We want collapse the folder hierarchy
            all_tar_files = input_dir.rglob("*.tar.gz")
            for archive in all_tar_files:
                output_path = output_dir / archive.stem
                my_tar = tarfile.open(archive)
                all_articles = [x for x in my_tar.getmembers() if x.isfile()]
                for article in all_articles:
                    output_path = output_dir / article.path.rpartition("/")[2]
                    f_in = my_tar.extractfile(article)
                    with open(output_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                my_tar.close()

        elif self.source == "pubmed":
            # .xml.gz
            all_zip_files = [p for p in input_dir.iterdir() if p.suffix == ".gz"]
            if not all_zip_files:
                raise ValueError("No zip files were found")

            for archive in all_zip_files:
                output_path = output_dir / archive.stem
                with gzip.open(archive, "rb") as f_in:
                    with open(output_path,"wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

        else:
            raise ValueError(f"Unsupported source {self.source}")




@requires(DownloadTask, UnzipTask)
class TopicExtractTask(luigi.Task):
    source = luigi.Parameter()

    def run(self):
        print(self.__class__.__name__)
        output_file = Path(self.output().path)
        output_file.touch()

    def requires(self):
        if self.source in {"pmc", "pubmed"}:
            return self.clone(UnzipTask)
        else:
            return self.clone(DownloadTask)

    def output(self):
        input_dir = self.input()[0]
        output_file = Path(input_dir.path).parent / "extraction_done.txt"

        return luigi.LocalTarget(str(output_file))

# @inherits(TopicExtractTask)
@requires(TopicExtractTask)
class TopicFilterTask(luigi.Task):
    filter_config = luigi.Parameter()

    def run(self):
        print(self.__class__.__name__)
        output_file = Path(self.output().path)
        output_file.touch()

    def output(self):
        output_file = Path(self.input().path).parent / "filtering_done.txt"

        return luigi.LocalTarget(str(output_file))

@requires(TopicFilterTask)
class ConvertPDFTask(luigi.Task):
    def run(self):
        print(self.__class__.__name__)
        output_file = Path(self.output().path)
        output_file.touch()

    def output(self):
        output_file = Path(self.input().path).parent / "converting_pdf_done.txt"

        return luigi.LocalTarget(str(output_file))


@inherits(ConvertPDFTask, TopicFilterTask)
# @requires(TopicFilterTask)
class ParseTask(luigi.Task):
    def run(self):
        print(self.__class__.__name__)

        output_file = Path(self.output().path)
        output_file.touch()

    def requires(self):
        if self.source == "arxiv":
            return self.clone(ConvertPDFTask)
        else:
            return self.clone(TopicFilterTask)

    def output(self):
        output_file = Path(self.input().path).parent / "parsing_done.txt"

        return luigi.LocalTarget(str(output_file))

@requires(ParseTask)
class AddTask(luigi.Task):
    def run(self):
        print(self.__class__.__name__)
        output_file = Path(self.output().path)
        output_file.touch()

    def output(self):
        output_file = Path(self.input().path).parent / "adding_done.txt"

        return luigi.LocalTarget(str(output_file))

@requires(AddTask)
class ListTask(ExternalProgramTask):
    capture_output = False
    def program_args(self):
        return ["ls", "-alh", "luigi/temp/"]


def run(
    *,
    source: str,
    from_month: str,
    filter_config: Path,
    output_dir: Path,
    db_url: str,
    db_type: str,
    mesh_topic_db: Path
) -> int:
    """Run overall pipeline.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    logger.info("Starting the overall pipeline")

    DownloadTask.capture_output = CAPTURE_OUTPUT 
    TopicExtractTask.capture_output = CAPTURE_OUTPUT 

    luigi.build(
        [
            AddTask(
                source=source,
                from_month=from_month,
                filter_config=str(filter_config),
                output_dir=str(output_dir),
            )
            # ListTask(source=source, from_month=from_month, filter_config=filter_config)
        ],
        log_level="INFO",
        # workers=0,
        local_scheduler=True,  # prevents the task already in progress errors
        # log_level="INFO"
    )

    return 0
