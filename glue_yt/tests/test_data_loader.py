from glue.core.data_factories import load_data
from glue_yt.data_loader import YTGlueData


def test_data_factory(simple_yt_dataset):
    data = load_data(simple_yt_dataset)
    assert isinstance(data, YTGlueData)
