from atp.tools.VectorStore import VectorStore
import pytest

@pytest.fixture
def store():
    return VectorStore("./tests/docs/md", "./tests/docs/store")

def test_sync_md(store: VectorStore):
    store.sync_md("./tests/docs/pdf")

def test_vec_sync(store: VectorStore):
    store.sync_store()