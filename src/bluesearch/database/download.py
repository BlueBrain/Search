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
import pathlib
from datetime import datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def get_days_list(
    start_date: datetime, end_date: datetime | None = None
) -> list[datetime]:
    """Retrieve list of days between a start date and an end date.

    Parameters
    ----------
    start_date
        Starting date.
    end_date
        Ending date. If None, today is considered as the ending date.

    Returns
    -------
    list of datetime
        List of all days between start date and end date included.
    """
    if end_date is None:
        end_date = datetime.today()

    delta = end_date - start_date

    days_list = []
    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
        days_list.append(day)

    return days_list


def get_pmc_urls(
    component: str, start_date: datetime, end_date: datetime | None = None
) -> list[str]:
    """Get list of all PMC incremental files to download.

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

    days_list = get_days_list(start_date=start_date, end_date=end_date)

    url_list = []
    for day in days_list:
        date_str = day.strftime("%Y-%m-%d")
        path_name = f"{component}_xml.incr.{date_str}.tar.gz"
        url = base_url + path_name
        url_list.append(url)

    return url_list


def download_pmc_articles(url_list: list[str], output_dir: Path) -> None:
    """Download PMC articles.

    Parameters
    ----------
    url_list
        List of URLs to query.
    output_dir
        Output directory to save the download.
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
