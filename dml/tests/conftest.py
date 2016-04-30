import pytest

import os
import shutil
import tempfile


@pytest.fixture()
def workspace(request):
    """Returns a path to a temporary directory for writing data."""
    test_workspace = tempfile.mkdtemp()

    def fin():
        if os.path.exists(test_workspace):
            shutil.rmtree(test_workspace)

    request.addfinalizer(fin)

    return test_workspace


@pytest.fixture(scope='module')
def test_dir():
    return os.path.dirname(__file__)


@pytest.fixture(scope='module')
def data_dir(test_dir):
    return os.path.join(test_dir, os.path.pardir, os.path.pardir,
                        'data', 'cqts')
