from bluesearch.k8s.create_indices import add_index, remove_index


def test_create_and_remove_index(index="test_index"):

    add_index(index)

    remove_index(index)
