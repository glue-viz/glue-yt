import astropy
import yt
import numpy as np

from yt.visualization.fixed_resolution import FixedResolutionBuffer
from glue.core import BaseCartesianData, DataCollection, ComponentID
from glue.core.coordinates import coordinates_from_wcs
from glue.config import data_factory
from glue.app.qt import GlueApplication


def _steps(slice):
    return int(np.ceil(1. * (slice.stop - slice.start) / slice.step))


class YTGlueData(BaseCartesianData):

    def __init__(self, ds):
        super(YTGlueData, self).__init__()
        self.ds = ds
        self.grid = ds.arbitrary_grid(
            ds.domain_left_edge, ds.domain_right_edge, (256,)*3)
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
        self._dds = (ds.domain_width / self.shape).to_value("code_length")
        self._left_edge = self.ds.domain_left_edge.to_value("code_length")
        self._right_edge = self.ds.domain_right_edge.to_value("code_length")

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
    def _get_loc(self, idx):
        return self._left_edge + idx*self._dds

    def _slice_args(self, view):
        index, coord = [(i, v) for i, v in enumerate(view)
                        if not isinstance(v, slice)][0]
        coord = self._get_loc(coord)[index]
        return index, coord

    def _frb_args(self, view, axis):
        ix = self.ds.coordinates.x_axis[axis]
        iy = self.ds.coordinates.y_axis[axis]
        sx = view[ix]
        sy = view[iy]
        l, r = sx.start, sx.stop
        b, t = sy.start, sy.stop
        w = _steps(sx)
        h = _steps(sy)
        bounds = (self._dds[ix]*l + self._left_edge[ix],
                  self._dds[ix]*r + self._left_edge[ix],
                  self._dds[iy]*b + self._left_edge[iy],
                  self._dds[iy]*t + self._left_edge[iy])
        return bounds, (h, w)

    def get_data(self, cid, view=None):
        nd = len([v for v in view if isinstance(v, slice)])
        field = tuple(cid.label.split())
        if nd == 2:
            axis, coord = self._slice_args(view)
            sl = self.ds.slice(axis, coord)
            frb = FixedResolutionBuffer(sl, *self._frb_args(view, axis))
            return frb[field].d.T
        else:
            return np.squeeze(self.grid[field][view].d)

    def compute_statistic(self, statistic, cid, subset_state=None, axis=None,
                          finite=True, positive=False, percentile=None,
                          view=None, random_subset=None):
        field = tuple(cid.label.split())
        if statistic == 'minimum':
            return float(self.region.min(field))
        elif statistic == 'maximum':
            return float(self.region.max(field))
        elif statistic == 'mean':
            return float(self.region.mean(field))
        elif statistic == 'median':
            return float(np.median(self.region[field]))
        elif statistic == 'sum':
            return float(self.region.sum(field))
        elif statistic == 'percentile':
            return float(np.percentile(self.region[field], percentile))

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
