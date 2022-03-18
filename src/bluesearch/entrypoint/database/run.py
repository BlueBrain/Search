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
import pathlib
import re
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
        "--final-task",
        type=str,
        choices=(
            "DownloadTask",
            "UnzipTask",
            "TopicExtractTask",
            "TopicFilterTask",
            "PerformFilteringTask",
            "ConvertPDFTask",
            "ParseTask",
            "AddTask",
        ),
        help="Final task of the luigi pipeline.",
    )
    parser.add_argument(
        "--luigi-config-path",
        type=Path,
        help="Path to Luigi configuration file. By default, "
        "luigi is looking into: /etc/luigi/luigi.cfg, luigi.cfg"
        "and the environment variable LUIGI_CONFIG_PATH."
        "If a path is specified, it is the one used.",
    )
    parser.add_argument(
        "--luigi-config-args",
        type=str,
        help="Comma separated key-value arguments for Luigi configuration, "
        "e.g. '--luigi-config GlobalParams.source:arxiv,"
        "DownloadTask.from-month:2021-04'. Overwrites the content of Luigi "
        "configuration file (see --luigi-config-path).",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Prints out a diagram of the pipeline without running it.",
    )

    return parser


BBS_BINARY = ["bbs_database"]
VERBOSITY = ["-v"]  # for the entrypoint subprocesses
CAPTURE_OUTPUT = False
IDENTIFIER = None  # make sure the same for all tasks


class GlobalParams(luigi.Config):
    """Global configuration."""

    source = luigi.Parameter()


class DownloadTask(ExternalProgramTask):
    """Download raw files.

    They will be stored in the `raw/` folder.
    """

    from_month = luigi.Parameter()
    output_dir = luigi.Parameter()
    identifier = luigi.OptionalParameter()

    def output(self) -> luigi.LocalTarget:
        """Define download folder."""
        global IDENTIFIER
        if self.identifier is not None:
            identifier = self.identifier

        else:
            if IDENTIFIER is None:
                today = datetime.today()
                identifier = f"{self.from_month}_{today.strftime('%Y-%m-%d')}"
                IDENTIFIER = identifier
            else:
                identifier = IDENTIFIER

        output_dir = Path(self.output_dir) / GlobalParams().source / identifier / "raw"

        return luigi.LocalTarget(str(output_dir))

    def program_args(self) -> list[str]:
        """Define subprocess arguments."""
        output_dir = self.output().path
        return [
            *BBS_BINARY,
            "download",
            *VERBOSITY,
            GlobalParams().source,
            self.from_month,
            output_dir,
        ]


class UnzipTask(ExternalProgramTask):
    """Unzip raw files (if necessary).

    Only applicable in case of `pmc`. The unzipped files
    are stored inside of `raw_unzipped`.
    """

    @staticmethod
    def requires() -> luigi.Task:
        """Define dependency."""
        return DownloadTask()

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
        if GlobalParams().source == "pmc":
            # .tar.gz
            # We want collapse the folder hierarchy
            all_tar_files = input_dir.rglob("*.tar.gz")
            for archive in all_tar_files:
                output_path = output_dir / archive.stem
                with tarfile.open(archive) as my_tar:
                    all_articles = [x for x in my_tar.getmembers() if x.isfile()]
                    for article in all_articles:
                        output_path = output_dir / article.path.rpartition("/")[2]
                        f_in = my_tar.extractfile(article)
                        with open(output_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)  # type: ignore

        else:
            raise ValueError(f"Unsupported source {GlobalParams().source}")


class TopicExtractTask(ExternalProgramTask):
    """Topic extraction.

    The input of this task is either `raw/` or `raw_unzipped/` depending
    on the source. The output is going to be a single file
    `topic_infos.jsonl`.
    """

    mesh_topic_db = luigi.Parameter()

    @staticmethod
    def requires() -> luigi.Task:
        """Define conditional dependencies."""
        if GlobalParams().source in {"pmc"}:
            return UnzipTask()
        else:
            return DownloadTask()

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
            GlobalParams().source,
            input_dir,
            output_dir,
        ]

        if GlobalParams().source in {"medrxiv", "biorxiv"}:
            command.extend(
                ["-R", "-m", r".*\.meca$"],
            )

        if GlobalParams().source in {"pmc", "pubmed"}:
            command.append(f"--mesh-topic-db={self.mesh_topic_db}")

        if GlobalParams().source == "pubmed":
            command.extend(
                ["-R", "-m", r".*\.xml\.gz$"],
            )

        return command


class TopicFilterTask(ExternalProgramTask):
    """Run topic filtering entrypoint.

    It inputs `topic_infos.jsonl` and `filter_config` (rules) and it
    generates a file `filtering.csv`.
    """

    filter_config = luigi.Parameter()

    @staticmethod
    def requires() -> luigi.Task:
        """Define dependency."""
        return TopicExtractTask()

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


class PerformFilteringTask(luigi.Task):
    """Create folder that only contains relevant articles.

    We only consider those articles that made it through the topic-filtering
    stage. The only input is the `filtering.csv`.
    """

    @staticmethod
    def requires() -> luigi.Task:
        """Define dependency."""
        return TopicFilterTask()

    def output(self) -> luigi.LocalTarget:
        """Define output folder."""
        output_dir = Path(self.input().path).parent / "filtered"

        return luigi.LocalTarget(str(output_dir))

    def run(self) -> None:
        """Create symlinks."""
        output_dir = Path(self.output().path)

        filtering = pd.read_csv(self.input().path)

        output_dir.mkdir(exist_ok=True)

        if GlobalParams().source == "pubmed":
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


class ConvertPDFTask(ExternalProgramTask):
    """Convert PDFs to XMLs.

    Assumes that there is a GROBID server up and running. Only necessary
    when `source=arxiv`. The output is the folder `converted_pdfs/`.
    """

    grobid_host = luigi.Parameter()
    grobid_port = luigi.IntParameter()

    @staticmethod
    def requires() -> luigi.Task:
        """Define dependency."""
        return PerformFilteringTask()

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


class ParseTask(ExternalProgramTask):
    """Parse articles.

    The input is all the articles inside of `filtered/` (or in case of
    `source="arxiv"` `converted_pdfs/`).
    """

    @staticmethod
    def requires() -> luigi.Task:
        """Define conditional dependencies."""
        if GlobalParams().source == "arxiv":
            return ConvertPDFTask()
        else:
            return PerformFilteringTask()

    def output(self) -> luigi.LocalTarget:
        """Define output folder."""
        output_file = Path(self.input().path).parent / "parsed"

        return luigi.LocalTarget(str(output_file))

    def program_args(self) -> list[str]:
        """Define subprocess arguments."""
        output_dir = Path(self.output().path)

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
        parser = source2parser[GlobalParams().source]

        command = [
            *BBS_BINARY,
            "parse",
            *VERBOSITY,
            parser,
            str(input_dir),
            str(output_dir),
        ]

        return command


class AddTask(ExternalProgramTask):
    """Add parsed articles to the database.

    This step is considered done if all articles inside of `parsed/` are
    already in the database.
    """

    db_url = luigi.Parameter()
    db_type = luigi.Parameter()

    @staticmethod
    def requires() -> luigi.Task:
        """Define dependency."""
        return ParseTask()

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
    dry_run: bool,
    final_task: str | None = None,
    luigi_config_path: Path | None = None,
    luigi_config_args: str | None = None,
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

    if luigi_config_path:
        if not pathlib.Path(luigi_config_path).exists():
            raise ValueError(
                f"The configuration path {luigi_config_path} " f"does not exist!"
            )

        config = luigi.configuration.get_config()
        config.add_config_path(luigi_config_path)
        config.reload()

    if luigi_config_args:
        config = luigi.configuration.get_config()
        for param in luigi_config_args.split(","):
            change = re.split(r"[.:]", param, maxsplit=3)
            config.set(*change)

    if final_task:
        final_task_call = globals()[final_task]
    else:
        final_task_call = AddTask

    if dry_run:
        print(print_tree(final_task_call(), last=False))
    else:
        luigi.build([final_task_call()])

    return 0
