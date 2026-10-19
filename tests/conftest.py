import pytest

from namegen.data import NamesDataset


@pytest.fixture(scope="session")
def dataset() -> NamesDataset:
    return NamesDataset()
