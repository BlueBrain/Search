"""Implementation of the MiningSchma class."""

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

import warnings

import pandas as pd


class MiningSchema:
    """The mining schema for the mining widget."""

    def __init__(self):
        self.columns = (
            "entity_type",
            "property",
            "property_type",
            "property_value_type",
            "ontology_source",
        )
        self.schema_df = pd.DataFrame(columns=self.columns)

    def add_entity(
        self,
        entity_type,
        property_name=None,
        property_type=None,
        property_value_type=None,
        ontology_source=None,
    ):
        """Add a new entity to the schema.

        A warning is issued for duplicate entities.

        Parameters
        ----------
        entity_type : str
            The entity type, for example "CHEMICAL".
        property_name: str, optional
            The property name, for example "isChiral".
        property_type : str, optional
            The property type, for example "ATTRIBUTE".
        property_value_type : str, optional
            The property value type, for example "BOOLEAN".
        ontology_source : str, optional
            The ontology source, for example "NCIT".
        """
        row = {
            "entity_type": entity_type,
            "property": property_name,
            "property_type": property_type,
            "property_value_type": property_value_type,
            "ontology_source": ontology_source,
        }
        # Make sure there are no duplicates to begin with
        self.schema_df = self.schema_df.drop_duplicates(ignore_index=True)
        # 'row' has type Dict[str, Any]. It is valid for append(). Ignoring the error.
        self.schema_df = self.schema_df.append(row, ignore_index=True)  # type: ignore[arg-type]  # noqa
        # If there are any duplicates at this point, then it must have
        # come from the appended row.
        if any(self.schema_df.duplicated()):
            self.schema_df = self.schema_df.drop_duplicates(ignore_index=True)
            warnings.warn("This entry already exists. No new entry was created.")

    def add_from_df(self, entity_df):
        """Add entities from a given dataframe.

        The data frame has to contain a column named "entity_type". Any
        columns matching the schema columns will be processed, all other
        columns will be ignored.

        Parameters
        ----------
        entity_df : pd.DataFrame
            The dataframe with new entities.
        """
        # The dataframe must contain the "entity_type" column
        if "entity_type" not in entity_df.columns:
            raise ValueError("Column named entity_type not found.")

        # Collect all other valid columns
        valid_columns = []
        for column in entity_df:
            if column in self.schema_df.columns:
                valid_columns.append(column)
            else:
                warnings.warn(f"No column named {column} was found.")

        # Add new data to the schema
        for _, row in entity_df[valid_columns].iterrows():
            self.add_entity(
                row["entity_type"],
                property_name=row.get("property"),
                property_type=row.get("property_type"),
                property_value_type=row.get("property_value_type"),
                ontology_source=row.get("ontology_source"),
            )

    @property
    def df(self):
        """Get a dataframe with all entities.

        Returns
        -------
        schema_df : pd.DataFrame
            The dataframe with all entities.
        """
        return self.schema_df.copy()
