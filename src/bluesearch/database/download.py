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
"""Facilities to download articles from different sources."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import boto3
import requests

logger = logging.getLogger(__name__)


def get_daterange_list(
        start_date: datetime, end_date: datetime | None = None, delta: str = "day",
) -> list[datetime]:
    """Retrieve list of dates between a start date and an end date (both inclusive).

    Parameters
    ----------
    start_date
        Starting date (inclusive).
    end_date
        Ending date (inclusive). If None, today is considered as the ending date.
    delta : {"day", "month"}
        Time difference between two consecutive dates.

    Returns
    -------
    list of datetime
        List of all days between start date and end date included.
    """
    if end_date is None:
        end_date = datetime.today()

    start_end_delta = end_date - start_date

    date_list = []

    if delta == "day":
        timedelta_ = timedelta(days=1)
        n_periods = start_end_delta.days

    elif delta == "month":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown delta: {delta}")

    for i in range(n_periods + 1):
        date = start_date + i * timedelta_
        date_list.append(date)

    return date_list


def generate_pmc_urls(
    component: str, start_date: datetime, end_date: datetime | None = None
) -> list[str]:
    """Generate the list of all PMC incremental files to download.

    Parameters
    ----------
    component : {"author_manuscript", "oa_comm", "oa_noncomm"}
        Part of the PMC to download.
    start_date
        Starting date to download the incremental files.
    end_date
        Ending date. If None, today is considered as the ending date.

    Returns
    -------
    list of str
        List of all the requests to make on PMC

    Raises
    ------
    ValueError
        If the chosen component does not exist on PMC.
    """
    base_url = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/"
    if component in {"oa_comm", "oa_noncomm"}:
        base_url += f"oa_bulk/{component}/xml/"
    elif component == "author_manuscript":
        base_url += "manuscript/xml/"
    else:
        raise ValueError(
            f"Unexcepted component {component}. "
            "Only {'author_manuscript', 'oa_comm', 'oa_noncomm'} are supported."
        )

    days_list = get_daterange_list(start_date=start_date, end_date=end_date)

    url_list = []
    for day in days_list:
        date_str = day.strftime("%Y-%m-%d")
        path_name = f"{component}_xml.incr.{date_str}.tar.gz"
        url = base_url + path_name
        url_list.append(url)

    return url_list


def get_pubmed_urls(
    start_date: datetime, end_date: datetime | None = None
) -> list[str]:
    """Get from the Internet the list of all PubMed incremental files to download.

    Parameters
    ----------
    start_date
        Starting date to download the incremental files (inclusive).
    end_date
        Ending date (inclusive). If None, today is considered as the ending date.

    Returns
    -------
    list of str
        List of all the PubMed urls.
    """
    if end_date is None:
        end_date = datetime.today()

    base_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles"
    response = requests.get(base_url)
    pattern = re.compile(r'<a href="(.*\.xml\.gz)">.*</a> *(\d{4}-\d{2}-\d{2})')
    urls = []

    for line in response.text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        file_name, date_str = match.groups()
        date = datetime.strptime(date_str, "%Y-%m-%d")
        if start_date <= date <= end_date:
            urls.append(f"{base_url}/{file_name}")

    return urls


def get_s3_urls(
    source: str,
    start_date: datetime,
    end_date: datetime | None = None
) -> dict[str, List[str]]:
    """Get S3 urls.

    We actually send a request to the AWS server and there is a charge.

    Parameters
    ----------
    source : {"medrxiv", "biorxiv"}
        Name of the source.
    start_date
        Starting date to download the incremental files (inclusive).
    end_date
        Ending date. If None, today is considered as the ending date (inclusive).

    Returns
    -------
    url_dict
        Keys represent different months. Values represent lists of the
        actual `.meca` files.

    """
    # checks
    if end_date is None:
        end_date = datetime.today()

    if source not in {"medrxiv", "biorxiv"}:
        raise ValueError(f"Unknown source: {source}")

    # generate November_2019, December_2019, ...

    # filtering objects using boto3
    raise NotImplementedError

def download_articles(url_list: list[str], output_dir: Path) -> None:
    """Download articles.

    Parameters
    ----------
    url_list
        List of URLs to query.
    output_dir
        Output directory to save the download. We assume that it already
        exists.
    """
    for url in url_list:
        logger.info(f"Requesting URL {url}")
        r = requests.get(url)
        _, _, file_name = url.rpartition("/")
        if not r.ok:
            logger.warning(
                f"URL {url} does not exist or "
                f"there was an issue when trying to retrieve it."
            )
            continue
        with open(output_dir / file_name, "wb") as f:
            f.write(r.content)
