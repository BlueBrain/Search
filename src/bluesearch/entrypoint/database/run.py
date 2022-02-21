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
import logging
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

import luigi
import pandas as pd
import sqlalchemy
from defusedxml import ElementTree
from defusedxml.ElementTree import tostring
from luigi.contrib.external_program import ExternalProgramTask
from luigi.tools.deps_tree import print_tree
from luigi.util import inherits, requires

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
    parser.add_argument(
        "--identifier",
        type=str,
        help="""Custom name of the identifier. If not specified, we use
        `from-month_today`
        """,
    )

    return parser


BBS_BINARY = ["bbs_database"]
VERBOSITY = ["-v"]  # for the entrypoint subprocesses
CAPTURE_OUTPUT = False
OUTPUT_DIR_RAW = None  # make sure the same datestamp for all tasks


class DownloadTask(ExternalProgramTask):
    """Download raw files.

    They will be stored in the `raw/` folder.
    """

    source = luigi.Parameter()
    from_month = luigi.Parameter()
    output_dir = luigi.Parameter()
    identifier = luigi.OptionalParameter()

    def output(self) -> luigi.LocalTarget:
        """Define download folder."""
        global OUTPUT_DIR_RAW
        if OUTPUT_DIR_RAW is None:
            today = datetime.today()
            if self.identifier is None:
                identifier = f"{self.from_month}_{today.strftime('%Y-%m-%d')}"
            else:
                identifier = self.identifier

            OUTPUT_DIR_RAW = Path(self.output_dir) / self.source / identifier / "raw"

        return luigi.LocalTarget(str(OUTPUT_DIR_RAW))

    def program_args(self) -> list[str]:
        """Define subprocess arguments."""
        output_dir = self.output().path
        return [
            *BBS_BINARY,
            "download",
            *VERBOSITY,
            self.source,
            self.from_month,
            output_dir,
        ]


@requires(DownloadTask)
class UnzipTask(ExternalProgramTask):
    """Unzip raw files (if necessary).

    Only applicable in case of `pmc`. The unzipped files
    are stored inside of `raw_unzipped`.
    """

    source = luigi.Parameter()

    def output(self) -> luigi.LocalTarget:
        """Define unzipping folder."""
        input_path = Path(self.input().path)
        output_dir = input_path.parent / "raw_unzipped"

        return luigi.LocalTarget(str(output_dir))

    def run(self) -> None:
        """Unzip."""
        input_dir = Path(self.input().path)  # raw
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
                        shutil.copyfileobj(f_in, f_out)  # type: ignore
                my_tar.close()

        else:
            raise ValueError(f"Unsupported source {self.source}")


@inherits(DownloadTask, UnzipTask)
class TopicExtractTask(ExternalProgramTask):
    """Topic extraction.

    The input of this task is either `raw/` or `raw_unzipped/` depending
    on the source. The output is going to be a single file
    `topic_infos.jsonl`.
    """

    source = luigi.Parameter()
    mesh_topic_db = luigi.Parameter()

    def requires(self) -> luigi.Task:
        """Define conditional dependencies."""
        if self.source in {"pmc"}:
            return self.clone(UnzipTask)
        else:
            return self.clone(DownloadTask)

    def output(self) -> luigi.LocalTarget:
        """Define output file path."""
        input_dir = self.input()
        output_file = Path(input_dir.path).parent / "topic_infos.jsonl"

        return luigi.LocalTarget(str(output_file))

    def program_args(self) -> list[str]:
        """Define subprocess arguments."""
        input_dir = self.input().path
        output_dir = self.output().path

        command = [
            *BBS_BINARY,
            "topic-extract",
            *VERBOSITY,
            self.source,
            input_dir,
            output_dir,
        ]

        if self.source in {"medrxiv", "biorxiv"}:
            command.extend(
                ["-R", "-m", r".*\.meca$"],
            )

        if self.source in {"pmc", "pubmed"}:
            command.append(f"--mesh-topic-db={self.mesh_topic_db}")

        return command


@requires(TopicExtractTask)
class TopicFilterTask(ExternalProgramTask):
    """Run topic filtering entrypoint.

    It inputs `topic_infos.jsonl` and `filter_config` (rules) and it
    generates a file `filtering.csv`.
    """

    filter_config = luigi.Parameter()

    def output(self) -> luigi.LocalTarget:
        """Define output file."""
        output_file = Path(self.input().path).parent / "filtering.csv"

        return luigi.LocalTarget(str(output_file))

    def program_args(self) -> list[str]:
        """Define subprocess arguments."""
        extracted_topics = self.input().path
        output_file = self.output().path

        command = [
            *BBS_BINARY,
            "topic-filter",
            *VERBOSITY,
            extracted_topics,
            self.filter_config,
            output_file,
        ]

        return command


@requires(TopicFilterTask)
class PerformFilteringTask(luigi.Task):
    """Create folder that only contains relevant articles.

    We only consider those articles that made it through the topic-filtering
    stage. The only input is the `filtering.csv`.
    """

    def output(self) -> luigi.LocalTarget:
        """Define output folder."""
        output_dir = Path(self.input().path).parent / "filtered"

        return luigi.LocalTarget(str(output_dir))

    def run(self) -> None:
        """Create symlinks."""
        output_dir = Path(self.output().path)

        filtering = pd.read_csv(self.input().path)

        output_dir.mkdir(exist_ok=True)

        if self.source == "pubmed":
            # Find all input files (.xml.gz)
            all_input_files = [Path(p) for p in filtering["path"].unique()]

            # Iteratively load each  of the files in memory
            for input_file in all_input_files:
                # Unzip it
                with gzip.open(input_file) as xml_stream:
                    article_set = ElementTree.parse(xml_stream)

                # Create a copy of the XML
                # article_set_copy = copy.deepcopy(article_set)
                root = article_set.getroot()

                # Find elements that were not accepted
                to_remove = filtering[
                    (filtering["path"] == str(input_file)) & (~filtering["accept"])
                ]
                article_nodes = root.findall("PubmedArticle")

                for eif in to_remove["element_in_file"].astype(int).tolist():
                    # Remove the corresponding <PubmedArticle> from the copy
                    root.remove(article_nodes[eif])

                # Store the copy with removed elements
                output_file = output_dir / input_file.name
                out_bytes = tostring(root)
                with gzip.open(output_file, "wb") as f:
                    f.write(out_bytes)

        else:
            accepted = pd.Series(filtering[filtering.accept].path.unique())

            def create_symlink(path):
                input_path = Path(path)
                output_path = output_dir / input_path.name
                output_path.symlink_to(input_path)

            accepted.apply(create_symlink)


@requires(PerformFilteringTask)
class ConvertPDFTask(ExternalProgramTask):
    """Convert PDFs to XMLs.

    Assumes that there is a GROBID server up and running. Only necessary
    when `source=arxiv`. The output is the folder `converted_pdfs/`.
    """

    grobid_host = luigi.Parameter()
    grobid_port = luigi.IntParameter()

    def program_args(self) -> list[str]:
        """Define subprocess arguments."""
        input_dir = Path(self.input().path).parent / "filtered"
        output_dir = self.output().path

        command = [
            *BBS_BINARY,
            "convert-pdf",
            *VERBOSITY,
            self.grobid_host,
            self.grobid_port,
            input_dir,
            f"--output-dir={output_dir}",
        ]

        return command

    def output(self) -> luigi.LocalTarget:
        """Define output folder."""
        output_file = Path(self.input().path).parent / "converted_pdfs"

        return luigi.LocalTarget(str(output_file))


@inherits(ConvertPDFTask, PerformFilteringTask)
class ParseTask(ExternalProgramTask):
    """Parse articles.

    The input is all the articles inside of `filtered/` (or in case of
    `source="arxiv"` `converted_pdfs/`).
    """

    def requires(self) -> luigi.Task:
        """Define conditional dependencies."""
        if self.source == "arxiv":
            return self.clone(ConvertPDFTask)
        else:
            return self.clone(PerformFilteringTask)

    def output(self) -> luigi.LocalTarget:
        """Define output folder."""
        output_file = Path(self.input().path).parent / "parsed"

        return luigi.LocalTarget(str(output_file))

    def program_args(self) -> list[str]:
        """Define subprocess arguments."""
        output_dir = Path(self.output().path)
        output_dir.mkdir(exist_ok=True)

        if (output_dir.parent / "converted_pdfs").exists():
            input_dir = output_dir.parent / "converted_pdfs"
        else:
            input_dir = output_dir.parent / "filtered"

        # Determine parser
        source2parser = {
            "arxiv": "tei-xml-arxiv",
            "biorxiv": "jats-meca",
            "medrxiv": "jats-meca",
            "pmc": "jats-xml",
            "pubmed": "pubmed-xml-set",
        }
        parser = source2parser[self.source]

        command = [
            *BBS_BINARY,
            "parse",
            *VERBOSITY,
            parser,
            str(input_dir),
            str(output_dir),
        ]

        return command


@requires(ParseTask)
class AddTask(ExternalProgramTask):
    """Add parsed articles to the database.

    This step is considered done if all articles inside of `parsed/` are
    already in the database.
    """

    db_url = luigi.Parameter()
    db_type = luigi.Parameter()

    def complete(self) -> bool:
        """Check if all articles inside of `parsed/` are in the database."""
        # If all the articles are inside
        if self.db_type == "sqlite":
            prefix = "sqlite:///"
        elif self.db_type == "postgres":
            prefix = "postgresql+pg8000://"
        else:
            raise ValueError

        engine = sqlalchemy.create_engine(f"{prefix}{self.db_url}")

        input_dir = Path(self.input().path)
        if not input_dir.exists():
            return False

        all_uids = [
            article.stem for article in input_dir.iterdir() if article.suffix == ".json"
        ]

        new_uids = []
        for uid in all_uids:
            query = sqlalchemy.text(
                "SELECT article_id from articles WHERE article_id = :uid"
            )
            res = engine.execute(query, uid=uid).fetchall()

            if not res:
                new_uids.append(uid)

        return not new_uids

    def program_args(self) -> list[str]:
        """Define subprocess arguments."""
        input_dir = Path(self.input().path)

        command = [
            *BBS_BINARY,
            "add",
            *VERBOSITY,
            self.db_url,
            input_dir,
            f"--db-type={self.db_type}",
        ]

        return command


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
    identifier: str | None,
) -> int:
    """Run overall pipeline.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    logger.info("Starting the overall pipeline")

    DownloadTask.capture_output = CAPTURE_OUTPUT
    UnzipTask.capture_output = CAPTURE_OUTPUT
    TopicExtractTask.capture_output = CAPTURE_OUTPUT
    TopicFilterTask.capture_output = CAPTURE_OUTPUT
    PerformFilteringTask.capture_output = CAPTURE_OUTPUT
    ConvertPDFTask.capture_output = CAPTURE_OUTPUT
    ParseTask.capture_output = CAPTURE_OUTPUT
    AddTask.capture_output = CAPTURE_OUTPUT

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
        identifier=identifier,
    )

    luigi_kwargs = {
        "tasks": [final_task],
        "log_level": "WARNING",
        "local_scheduler": True,
    }
    if dry_run:
        print(print_tree(final_task, last=False))
    else:

        luigi.build(**luigi_kwargs)

    return 0
