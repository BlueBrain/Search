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
import pandas as pd
import sqlalchemy
from luigi.util import inherits, requires
from luigi.contrib.external_program import ExternalProgramTask
from luigi.tools.deps_tree import print_tree

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
        "--source",
        required=True,
        type=str,
        choices=[member.value for member in ArticleSource],
        help="Source of the articles.",
    )
    parser.add_argument(
        "--from-month",
        required=True,
        type=str,
        help="The starting month (included) for the download in format YYYY-MM. "
        "All papers from the given month until today will be downloaded.",
    )
    parser.add_argument(
        "--filter-config",
        required=True,
        type=Path,
        help="""
        Path to a .JSONL file that defines all the rules for filtering.
        """,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="""
        Path to the output folder. All the results stored under
        `output_dir/source/date` where date is concatenation of the
        `from_month` and the day of execution of this command.
        """,
    )
    parser.add_argument(
        "--db-url",
        required=True,
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
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Prints out a diagram of the pipeline without running it.",
    )
    parser.add_argument(
        "--grobid-host",
        type=str,
        help="The host of the GROBID server.",
    )
    parser.add_argument(
        "--grobid-port",
        type=int,
        help="The port of the GROBID server.",
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




@inherits(DownloadTask, UnzipTask)
class TopicExtractTask(ExternalProgramTask):
    source = luigi.Parameter()
    mesh_topic_db = luigi.Parameter()

    def requires(self):
        if self.source in {"pmc", "pubmed"}:
            return self.clone(UnzipTask)
        else:
            return self.clone(DownloadTask)

    def output(self):
        input_dir = self.input()
        output_file = Path(input_dir.path).parent / "topic_infos.jsonl"

        return luigi.LocalTarget(str(output_file))


    def program_args(self):
        input_dir = self.input().path
        output_dir = self.output().path

        command = [
            BBS_BINARY, "topic-extract", "-v", self.source, input_dir, output_dir, 
        ]
 
        if self.source in {"pmc", "pubmed"}:
            command.append(f"--mesh-topic-db={self.mesh_topic_db}")

        return command


@requires(TopicExtractTask)
class TopicFilterTask(ExternalProgramTask):
    filter_config = luigi.Parameter()

    def output(self):
        output_file = Path(self.input().path).parent / "filtering.csv"

        return luigi.LocalTarget(str(output_file))

    def program_args(self):
        extracted_topics = self.input().path
        output_file = self.output().path

        command = [
            BBS_BINARY, "topic-filter", "-v", extracted_topics, self.filter_config, output_file, 
        ]
 
        return command


@requires(TopicFilterTask)
class CreateSymlinksTask(luigi.Task):
    def output(self):
        output_dir = Path(self.input().path).parent / "filtered"

        return luigi.LocalTarget(str(output_dir))

    def run(self):
        output_dir = Path(self.output().path)
        filtering_path = Path(self.input().path)
        input_dir = output_dir.parent / "raw_unzipped" 

        if (output_dir.parent / "raw_unzipped").exists():
            input_dir = output_dir.parent / "raw_unzipped"
        else:
            input_dir = output_dir.parent / "raw"

        filtering = pd.read_csv(filtering_path)
        accepted = filtering[filtering.accept].path

        def create_symlink(path):
            input_path = Path(path)
            output_path = output_dir / input_path.name
            output_path.symlink_to(input_path)

        output_dir.mkdir(exist_ok=True)

        accepted.apply(create_symlink)




@requires(CreateSymlinksTask)
class ConvertPDFTask(ExternalProgramTask):
    grobid_host = luigi.Parameter()
    grobid_port = luigi.Parameter()


    def program_args(self):
        input_dir = Path(self.input().path).parent / "filtered"
        output_dir = self.output().path

        command = [
            BBS_BINARY,
            "convert-pdf",
            "-v",
            self.grobid_host,
            self.grobid_port, 
            input_dir,
            f"--output-dir={output_dir}",
        ]
 
        return command

    def output(self):
        output_file = Path(self.input().path).parent / "converted_pdfs"

        return luigi.LocalTarget(str(output_file))


@inherits(ConvertPDFTask, CreateSymlinksTask)
class ParseTask(ExternalProgramTask):
    def requires(self):
        if self.source == "arxiv":
            return self.clone(ConvertPDFTask)
        else:
            return self.clone(TopicFilterTask)

    def output(self):
        output_file = Path(self.input().path).parent / "parsed"

        return luigi.LocalTarget(str(output_file))

    def program_args(self):
        output_dir = Path(self.output().path)
        output_dir.mkdir(exist_ok=True)


        if (output_dir.parent / "converted_pdfs").exists():
            input_dir = output_dir.parent / "converted_pdfs"
        else:
            input_dir = output_dir.parent / "filtered"

        # Determine parser
        source2parser = {
            "arxiv": "tei-xml-arxiv",
            "biorxiv": "jatx-xml",
            "medrxiv": "jatx-xml",
            "pmc": "jatx-xml",
            "pubmed": "pubmed-xml",
        }
        parser = source2parser[self.source]

        command = [
            BBS_BINARY,
            "parse",
            "-v",
            parser,
            input_dir, 
            output_dir,
        ]
 
        return command


@requires(ParseTask)
class AddTask(ExternalProgramTask):
    db_url = luigi.Parameter()
    db_type = luigi.Parameter()

    def complete(self):
        # If all the articles are inside
        if self.db_type == "sqlite":
            prefix = "sqlite:///"
        elif self.db_type == "postgres":
            prefix = "postgresql+pg8000://"
        else:
            raise ValueError

        engine = sqlalchemy.create_engine(f"{prefix}{self.db_url}")

        input_dir = Path(self.input().path)
        all_uids = [article.stem for article in input_dir.iterdir() if article.suffix == ".json"]

        new_uids = []
        for uid in all_uids:
            query = "SELECT article_id from articles WHERE article_id = ?"
            res = engine.execute(query, (uid,)).fetchall()

            if not res:
                new_uids.append(uid)

        return not new_uids


    def program_args(self):
        input_dir = Path(self.input().path)


        command = [
            BBS_BINARY,
            "add",
            self.db_url,
            input_dir,
            "-v",
            f"--db-type={self.db_type}",
        ]
 
        return command



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
    mesh_topic_db: Path | None,
    dry_run: bool,
    grobid_host: str | None,
    grobid_port: int | None,
) -> int:
    """Run overall pipeline.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    logger.info("Starting the overall pipeline")

    DownloadTask.capture_output = CAPTURE_OUTPUT 
    TopicExtractTask.capture_output = CAPTURE_OUTPUT 

    final_task = AddTask(
        source=source,
        from_month=from_month,
        filter_config=str(filter_config),
        output_dir=str(output_dir),
        mesh_topic_db=str(mesh_topic_db),
        grobid_host=grobid_host,
        grobid_port=grobid_port,
        db_url=db_url,
        db_type=db_type,
    )

    luigi_kwargs = {
        "tasks": [final_task],
        "log_level": "DEBUG",
        "local_scheduler": True,
    }
    if dry_run:
        print(print_tree(final_task, last=False))
    else:

        luigi.build(**luigi_kwargs)

    return 0
