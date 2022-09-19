import pytest

from bluesearch.k8s.create_indices import add_index, remove_index


@pytest.mark.parametrize(
    "indices",
    ["test_single", ["test_multiple_0", "test_multiple_1", "test_multiple_2"]],
)
def test_create_and_remove_indices(indices):

    add_index(indices)

    remove_index(indices)
