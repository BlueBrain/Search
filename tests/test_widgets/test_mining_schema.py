import pytest

from bbsearch.widgets import MiningSchema


def test_add_entity():
    mining_schema = MiningSchema()

    # Test adding entities
    mining_schema.add_entity(
        "CHEMICAL",
        property_name="isChiral",
        property_type="ATTRIBUTE",
        property_value_type="BOOLEAN",
        ontology_source="NCIT",
    )
    mining_schema.add_entity("DRUG")
    assert len(mining_schema.schema_df) == 2

    # Test warning upon adding a duplicate entity
    with pytest.warns(UserWarning, match=r"already exists"):
        mining_schema.add_entity("DRUG")


def test_df(mining_schema_df):
    # We won't be testing for duplicates in this test
    mining_schema_df = mining_schema_df.drop_duplicates(ignore_index=True)

    # Test adding from a dataframe
    mining_schema = MiningSchema()
    mining_schema.add_from_df(mining_schema_df)
    # Make sure a copy is returned
    assert mining_schema.df is not mining_schema.schema_df
    # Check that all data was added
    assert mining_schema.df.equals(mining_schema_df)

    # Test missing entity_type
    wrong_schema_df = mining_schema_df.copy()
    wrong_schema_df.drop("entity_type", axis=1, inplace=True)
    mining_schema = MiningSchema()
    with pytest.raises(ValueError, match=r"entity_type.* not found"):
        mining_schema.add_from_df(wrong_schema_df)

    # Test ignoring unknown columns
    schema_df_new = mining_schema_df.drop_duplicates().copy()
    schema_df_new["unknown_column"] = [i for i in range(len(schema_df_new))]
    mining_schema = MiningSchema()
    with pytest.warns(UserWarning, match=r"column.* unknown_column"):
        mining_schema.add_from_df(schema_df_new)
    # Check that all data was added and the unknown columns was ignored
    assert mining_schema.df.equals(mining_schema_df)
