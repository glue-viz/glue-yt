import astropy
import yt
import numpy as np

from glue.core import BaseCartesianData, DataCollection, ComponentID
from glue.core.coordinates import coordinates_from_wcs
from glue.config import data_factory
from glue.app.qt import GlueApplication


class YTGlueData(BaseCartesianData):

    def __init__(self, ds, units=None, level_decrement=None):
        super(YTGlueData, self).__init__()
        self.ds = ds
        if units is None:
            self.units = ds.get_smallest_appropriate_unit(ds.domain_width[0])
        else:
            self.units = units
        if level_decrement is None:
            level_decrement = 0
        self.level_decrement = level_decrement
        self.region = ds.all_data()
        self.cids = [
            ComponentID('{} {}'.format(*f.name), parent=self)
            for f in ds.fields.gas]
        self._dds = (ds.domain_width / self.shape).to_value(self.units)
        self._left_edge = self.ds.domain_left_edge.to_value(self.units)
        self._right_edge = self.ds.domain_right_edge.to_value(self.units)
        w = astropy.wcs.WCS(naxis=3)
        c = 0.5*(self._left_edge + self._right_edge)
        w.wcs.cunit = [self.units]*3
        w.wcs.crpix = 0.5*(np.array(self.shape)+1)
        w.wcs.cdelt = self._dds
        w.wcs.crval = c
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
            i = self.ds.index.max_level-self.level_decrement
            refine_factor = self.ds.refine_by**i
            shp = refine_factor * self.ds.domain_dimensions
            self._shape = tuple(shp.astype("int"))
        return self._shape

    def get_kind(self, cid):
        return 'numerical'

    def get_mask(self, subset_state, view=None):
        breakpoint()

    def _get_loc(self, global_index, ax=None):
        ret = self._left_edge + global_index*self._dds
        if ax is None:
            return ret
        return ret[ax]

    """
    def get_data(self, cid, view=None):
        print("hello")
        if view is None:
            nd = self.ndim
        else:
            nd = len([v for v in view if isinstance(v, slice)])
        field = tuple(cid.label.split())
        if nd == 2:
            print("I'm in get_data")
            axis, coord = self._slice_args(view)
            sl = self.ds.slice(axis, coord)
            frb = FixedResolutionBuffer(sl, *self._frb_args(view, axis))
            return frb[field].d.T
        elif nd == 3:
            le = []
            re = []
            shape = []
            for i, v in enumerate(view):
                le.append(self._get_loc(v.start, i))
                re.append(self._get_loc(v.stop, i))
                shape.append((v.stop - v.start)//v.step)
            ag = self.ds.arbitrary_grid(le, re, shape)
            return ag[field].d
    """

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

    def compute_histogram(self, cids, weights=None, range=None, bins=None, log=None,
                          subset_state=None):
        fields = [tuple(cid.label.split()) for cid in cids]
        if weights is not None:
            weights = tuple(weights.label.split())
        profile = self.region.profile(fields, ['ones'], n_bins=bins[0],
            extrema={fields[0]: range[0]}, logs={fields[0]: log[0]},
            weight_field=weights)
        return profile['ones'].d

    def compute_fixed_resolution_buffer(self, bounds, target_data=None, 
                                        target_cid=None, subset_state=None, 
                                        broadcast=True, cache_id=None):
        print("I'm in cfrb")
        view = []
        print(bounds)
        for i, bound in enumerate(bounds):
            if isinstance(bound, tuple):
                start = (self._get_loc(bound[0], ax=i), self.units)
                stop = (self._get_loc(bound[1], ax=i), self.units)
                view.append(slice(start, stop, bound[2]*1j))
            else:
                view.append((self._get_loc(bound, ax=i), self.units))
        print(view)
        field = tuple(target_cid.label.split())
        return self.ds.r[view[0],view[1],view[2]][field].d


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
    ds = yt.load('GasSloshing/sloshing_nomag2_hdf5_plt_cnt_0150')
    def logdensity(field, data):
        return np.log10(data['gas', 'density'])
    ds.add_field(('gas', 'logdensity'), function=logdensity, units='',
                 sampling_type='cell')
    def logtemperature(field, data):
        return np.log10(data['gas', 'temperature'])
    ds.add_field(('gas', 'logtemperature'), function=logtemperature, units='',
                 sampling_type='cell')
    d1 = YTGlueData(ds, level_decrement=2)
    dc = DataCollection([d1])
    ga = GlueApplication(dc)
    #viewer = ga.new_data_viewer(VispyVolumeViewer)
    #viewer.add_data(d1)
    ga.start(maximized=False)
