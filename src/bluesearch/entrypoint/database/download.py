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
"""Download articles from different sources."""
import argparse
import getpass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import chain
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_to_datetime(s: str) -> datetime:
    """Try to convert a string to a datetime.

    Parameters
    ----------
    s
        String to be check as a valid date.

    Returns
    -------
    datetime
        The date specified in the input string.

    Raises
    ------
    ArgumentTypeError
        When the specified string has not a valid date format.
    """
    try:
        return datetime.strptime(s, "%Y-%m")
    except ValueError:
        msg = f"{s} is not a valid date"
        raise argparse.ArgumentTypeError(msg)


def init_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initialise the argument parser for the download subcommand.

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
    parser.description = "Download articles."

    parser.add_argument(
        "source",
        type=str,
        choices=("arxiv", "biorxiv", "medrxiv", "pmc", "pubmed"),
        help="Source of the download.",
    )
    parser.add_argument(
        "from_month",
        type=convert_to_datetime,
        help="The starting month (included) for the download in format YYYY-MM. "
        "All papers from the given month until today will be downloaded.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save the downloaded articles.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="""
        Display requests for the download.
        """,
    )
    return parser


def run(source: str, from_month: datetime, output_dir: Path, dry_run: bool) -> int:
    """Download articles of a source from a specific date.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    import boto3

    from bluesearch.database.download import (
        download_articles,
        download_s3_articles,
        generate_pmc_urls,
        get_pubmed_urls,
        get_s3_urls,
    )

    if source == "pmc":
        url_dict = {}
        for component in {"author_manuscript", "oa_comm", "oa_noncomm"}:
            url_dict[component] = generate_pmc_urls(component, from_month)

        if dry_run:
            for component, url_list in url_dict.items():
                print(f"URL requests from {component}:")
                print(*url_list, sep="\n")
            return 0

        logger.info("Start downloading PMC papers.")
        for component, url_list in url_dict.items():
            component_dir = output_dir / component
            logger.info(
                f"Start downloading {component} in {component_dir.resolve().as_uri()}"
            )
            component_dir.mkdir(exist_ok=True, parents=True)
            download_articles(url_list, component_dir)
        return 0
    elif source == "pubmed":
        url_list = get_pubmed_urls(from_month)
        if dry_run:
            print("URL requests from:")
            print(*url_list, sep="\n")
            return 0

        logger.info("Start downloading PubMed papers.")
        output_dir.mkdir(exist_ok=True, parents=True)
        download_articles(url_list, output_dir)
        return 0
    elif source in {"biorxiv", "medrxiv"}:

        key_id = getpass.getpass("aws_access_key_id: ")
        secret_access_key = getpass.getpass("aws_secret_access_key: ")

        session = boto3.Session(
            aws_access_key_id=key_id,
            aws_secret_access_key=secret_access_key,
        )
        resource = session.resource("s3")
        bucket = resource.Bucket(f"{source}-src-monthly")

        url_dict = get_s3_urls(bucket, from_month)

        if dry_run:
            for month, url_list in url_dict.items():
                print(f"Month: {month}")
                print(*url_list, sep="\n")
            return 0

        logger.info(f"Start downloading {source} papers.")
        download_s3_articles(bucket, url_dict, output_dir)
        return 0
    elif source == "arxiv":
        logger.info("Loading libraries")
        from google.cloud.storage import Client

        from bluesearch.database.download import download_gcs_blob, get_gcs_urls

        client = Client.create_anonymous_client()
        bucket = client.bucket("arxiv-dataset")
        if from_month < datetime(2007, 4, 1):
            logger.error(
                "The papers from before April 2007 follow a different format "
                "and can't be downloaded. Please contact the developers if you "
                "need them. To process please re-run the command with a "
                "different starting month."
            )
            return 1

        logger.info("Collecting download URLs")
        blobs_by_month = get_gcs_urls(bucket, from_month)

        if dry_run:
            print("The following items will be downloaded:")
            for month, month_blobs in blobs_by_month.items():
                print(f"Month: {month}")
                for blob in month_blobs:
                    print(blob.name)
            return 0

        def progress_info(n_jobs, n_bytes_):
            logger.info(f"{n_jobs} download jobs submitted ({n_bytes_:,d} bytes)")

        job_names = {}
        n_blobs = 0
        n_bytes = 0
        # The max_workers parameter already has a reasonable default if
        # not specified. See python docs for ThreadPoolExecutor.
        with ThreadPoolExecutor() as executor:
            logger.info("Submitting download jobs to workers")
            for blob in chain(*blobs_by_month.values()):
                future = executor.submit(
                    download_gcs_blob,
                    blob,
                    output_dir,
                    flatten=True,
                )
                job_names[future] = blob.name
                n_blobs += 1
                n_bytes += blob.size or 0
                if n_blobs % 100 == 0:
                    progress_info(n_blobs, n_bytes)
            progress_info(n_blobs, n_bytes)
            logger.info("Waiting for the downloads to finish (may take a while)")
            for future in as_completed(job_names):
                job_name = job_names[future]
                exc = future.exception()
                if exc:
                    logger.error("The job %s failed, reason: %s", job_name, exc)
                else:
                    logger.debug("The job %s succeeded.", job_name)
            logger.info("Finished downloading")
    else:
        logger.error(f"The source type {source!r} is not implemented yet")
        return 1

    return 0
