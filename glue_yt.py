import astropy
import yt
import numpy as np

from glue.core import BaseCartesianData, DataCollection, ComponentID
from glue.core.coordinates import coordinates_from_wcs
from glue.config import data_factory
from glue.app.qt import GlueApplication


class YTGlueData(BaseCartesianData):

    def __init__(self, ds):
        super(YTGlueData, self).__init__()
        self.ds = ds
        self.grid = ds.arbitrary_grid(
            ds.domain_left_edge, ds.domain_right_edge, (256,)*3)
        self.region = ds.box(ds.domain_left_edge, ds.domain_right_edge)
        self.region = ds.all_data()
        self.cids = [
            ComponentID('{} {}'.format(*f.name), parent=self)
            for f in ds.fields.gas]
        w = astropy.wcs.WCS(naxis=3)
        c = 0.5*(self.grid.left_edge + self.grid.right_edge)
        c = c.in_units('kpc')
        w.wcs.cunit = [str(c.units)]*3
        w.wcs.crpix = 0.5*(np.array(self.grid.shape)+1)
        w.wcs.cdelt = self.grid.dds.in_units('kpc').d
        w.wcs.crval = c.d
        self.coords = coordinates_from_wcs(w)
        wcids = []
        for i in range(self.ndim):
            label = self.coords.axis_label(i)
            wcids.append(ComponentID(label, parent=self))
        self._world_component_ids = wcids

    @property
    def label(self):
        return str(ds)

    @property
    def main_components(self):
        return self.cids

    @property
    def world_component_ids(self):
        return self._world_component_ids

    _shape = None
    @property
    def shape(self):
        if self._shape is None:
            refine_factor = self.ds.refine_by**self.ds.index.max_level
            shp = refine_factor * self.ds.domain_dimensions
            self._shape = tuple(shp.astype("int"))
        return self._shape

    def get_kind(self, cid):
        return 'numerical'

    def get_mask(self, subset_state, view=None):
        breakpoint()

    def get_data(self, cid, view=None):
        field = tuple(cid.label.split())
        return np.squeeze(self.grid[field][view].d)

    def compute_statistic(self, statistic, cid, subset_state=None, axis=None,
                          finite=True, positive=False, percentile=None,
                          view=None, random_subset=None):
        field = tuple(cid.label.split())
        if statistic == 'minimum':
            return float(self.grid[field].min(axis=axis))
        elif statistic == 'maximum':
            return float(self.grid[field].max(axis=axis))
        elif statistic == 'mean':
            return float(np.mean(self.grid[field], axis=axis))
        elif statistic == 'median':
            return float(np.median(self.grid[field], axis=axis))
        elif statistic == 'sum':
            return float(np.sum(self.grid[field], axis=axis))
        elif statistic == 'percentile':
            return float(np.percentile(self.grid[field], percentile, axis=axis))

    def compute_histogram(self, cids, range=None, bins=None, log=None,
                          subset_state=None):
        fields = [tuple(cid.label.split()) for cid in cids]
        profile = yt.create_profile(
            self.region, fields, ['ones'], n_bins=bins[0],
            extrema={fields[0]: range[0]}, logs={fields[0]: log[0]},
            weight_field=None)
        return profile['ones']


def is_yt_dataset(filename):
    try:
        yt.load(filename)
    except Exception:
        return False
    return True


@data_factory('yt dataset', is_yt_dataset)
def read_yt(filename):
    ds = yt.load(filename)
    return YTGlueData(ds)


if __name__ == "__main__":
    ds = yt.load('Enzo_64/DD0043/data0043')
    def logdensity(field, data):
        return np.log10(data['gas', 'density'])
    ds.add_field(('gas', 'logdensity'), function=logdensity, units='',
                 sampling_type='cell')
    def logtemperature(field, data):
        return np.log10(data['gas', 'temperature'])
    ds.add_field(('gas', 'logtemperature'), function=logtemperature, units='',
                 sampling_type='cell')
    d1 = YTGlueData(ds)
    d2 = YTGlueData(ds)
    dc = DataCollection([d1, d2])
    ga = GlueApplication(dc)
    ga.start(maximized=False)
