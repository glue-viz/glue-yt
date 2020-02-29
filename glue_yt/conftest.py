import tarfile
from urllib.request import urlretrieve

import pytest

from glue_yt import setup


def pytest_configure(config):
    setup()


@pytest.fixture
def simple_yt_dataset(request, tmp_path):
    urlretrieve('http://yt-project.org/data/MHDCTOrszagTang.tar.gz',
                filename=tmp_path / 'MHDCTOrszagTang.tar.gz')
    tf = tarfile.open(tmp_path / 'MHDCTOrszagTang.tar.gz')
    tf.extractall(tmp_path)
    return str(tmp_path / 'MHDCTOrszagTang' / 'DD????' / 'data????')
