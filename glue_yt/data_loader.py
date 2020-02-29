import astropy
import yt
from yt.data_objects.time_series import DatasetSeries
import numpy as np

from yt.visualization.fixed_resolution import FixedResolutionBuffer
from glue.core import BaseCartesianData, DataCollection, ComponentID
from glue.core.coordinates import coordinates_from_wcs
from glue.config import data_factory
from glue.app.qt import GlueApplication
from glue.core import Subset
from re import sub

def cid_to_field(cid):
    if cid is None:
        return
    if cid.label.startswith("World"):
        ax = int(cid.label.split()[1])
        return "index", "xyz"[ax]
    if cid.label.startswith("Pixel"):
        ax = int(cid.label.split()[-1])
        return "index", "pixel_{}".format("xyz"[ax])
    return tuple(cid.label.replace('"','').split(","))


class YTGlueData(BaseCartesianData):

    def __init__(self, ds_all, units=None):
        super(YTGlueData, self).__init__()
        self.ds_all = ds_all
        self.ds = ds_all[0]
        self._current_step = 0
        if units is None:
            self.units = self.ds.get_smallest_appropriate_unit(self.ds.domain_width[0])
        else:
            self.units = units
        self.region = self.ds.all_data()
        self.cids = [
            ComponentID('"{}","{}"'.format(*f.name), parent=self)
            for f in self.ds.fields.gas]
        self._dds = (self.ds.domain_width / self.shape).d
        self._left_edge = self.ds.domain_left_edge.d
        self._right_edge = self.ds.domain_right_edge.d
        w = astropy.wcs.WCS(naxis=3)
        c = 0.5*(self.ds.domain_left_edge + self.ds.domain_right_edge)
        w.wcs.cunit = [self.units]*3
        w.wcs.crpix = 0.5*(np.array(self.shape)+1)
        w.wcs.cdelt = self.ds.arr(self._dds, "code_length").to_value(self.units)
        w.wcs.crval = c.to_value(self.units)
        self.wcs = w
        self.coords = coordinates_from_wcs(w)
        wcids = []
        for i in range(self.ndim):
            label = self.coords.axis_label(i)
            wcids.append(ComponentID(label, parent=self))
        self._world_component_ids = wcids
        for i, ax in enumerate("xyz"):
            def _pixel_c(field, data):
                return self._get_pix(data["index", ax].d, ax=i)
            self.ds.add_field(("index", "pixel_{}".format(ax)), _pixel_c, units="")

    @property
    def current_step(self):
        return self._current_step

    @current_step.setter
    def current_step(self, value):
        self._current_step = value
        self.ds = self.ds_all[value]
        self.region = self.ds.all_data()

    @property
    def label(self):
        return str(self.ds)

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
            if hasattr(self.ds.index, "max_level"):
                i = self.ds.index.max_level
                refine_factor = self.ds.refine_by**i
                shp = refine_factor * self.ds.domain_dimensions
                self._shape = tuple(shp.astype("int"))
            else:
                self._shape = (512,)*3
        return self._shape

    def get_kind(self, cid):
        return 'numerical'

    def get_mask(self, subset_state, view=None):
        breakpoint()

    def _get_loc(self, global_idx, ax=None):
        ret = self._left_edge + (global_idx+0.5)*self._dds
        if ax is None:
            return ret
        return ret[ax]

    def _get_pix(self, loc, ax):
        return (loc-self._left_edge[ax])/self._dds[ax]-0.5

    def _get_loc_wcs(self, idx, ax):
        w = self.wcs.wcs
        ret = w.cdelt[ax]*(idx+1-w.crpix[ax])+w.crval[ax]
        return ret

    def get_data(self, cid, view=None):
        if view is not None:
            for i, v in enumerate(view):
                if isinstance(v, slice):
                    break
            if cid.label.startswith("World"):
                return self._get_loc_wcs(np.arange(self.shape[i]), i)
            elif cid.label.startswith("Pixel"):
                return np.arange(self.shape[i])
        if self.size > 100000:
            return np.array([self.compute_statistic("minimum", cid),
                             self.compute_statistic("maximum", cid)])
        else:
            bounds = [(-0.5, s-0.5, s) for s in self.shape]
            return self.compute_fixed_resolution_buffer(bounds, target_cid=cid)

    def _get_subset_region(self, subset_state=None):
        print(str(subset_state) + ' in GSR')
        if subset_state is not None:
            try:
                roi = subset_state.roi
                reg = self.region.include_inside(field[1], subset_state.lo, subset_state.hi)
            except AttributeError:
                field = cid_to_field(subset_state.att)
                reg = self.region.include_inside(field[1], subset_state.lo, subset_state.hi)
        else:
            reg = self.region
        return reg

    def compute_statistic(self, statistic, cid, subset_state=None, axis=None,
                          finite=True, positive=False, percentile=None,
                          view=None, random_subset=None):
        axes = {(1, 2): 0,
                (0, 2): 1,
                (0, 1): 2}
        field = cid_to_field(cid)

        #Get the subset info
        print('Calling GSR in compute_statistic')
        reg = self._get_subset_region(subset_state)

        if axis is None:
            # Compute statistic for all data
            if statistic == 'minimum':
                return float(reg.min(field))
            elif statistic == 'maximum':
                return float(reg.max(field))
            elif statistic == 'mean':
                return float(reg.mean(field))
            elif statistic == 'median':
                return float(np.median(reg[field]))
            elif statistic == 'sum':
                return float(reg.sum(field))
            elif statistic == 'percentile':
                return float(np.percentile(reg[field], percentile))
        else:
            nax = self.shape[axes[axis]]
            stat = np.zeros(nax)
            ns = nax // 8
            ei = 0
            bounds = [(-0.5, s-0.5, s) for s in self.shape]
            for i in range(8):
                si = ei
                ei = min(si+ns, self.shape[axes[axis]])
                bounds[axes[axis]] = (-0.5+si, ei-0.5, ns)
                cg = self.compute_fixed_resolution_buffer(bounds, target_cid=cid)
                # Compute statistic for a slice along axis tuple
                if statistic == 'minimum':
                    stat[si:ei] = cg.min(axis=axis)
                elif statistic == 'maximum':
                    stat[si:ei] = cg.max(axis=axis)
                elif statistic == 'mean':
                    stat[si:ei] = cg.mean(axis=axis)
                elif statistic == 'median':
                    stat[si:ei] = np.median(cg, axis=axis)
                elif statistic == 'sum':
                    stat[si:ei] = cg.sum(axis=axis)
                elif statistic == 'percentile':
                    stat[si:ei] = np.percentile(cg, percentile, axis=axis)
            return stat

    def compute_histogram(self, cids, weights=None, range=None, bins=None, log=None,
                          subset_state=None):
        # We use a yt profile over "ones" to make the histogram
        bin_fields = [cid_to_field(cid) for cid in cids]
        print('Calling GSR in compute_histogram')
        reg = self._get_subset_region(subset_state)
        if weights is None:
            field = "ones"
        else:
            field = cid_to_field(weights)
        extrema = {fd: r for fd, r in zip(bin_fields, range)}
        logs = {fd: l for fd, l in zip(bin_fields, log)}
        units = {fd: self.units for fd in bin_fields if fd[1] in "xyz"}
        profile = reg.profile(bin_fields, field, n_bins=bins,
            extrema=extrema, logs=logs, units=units, weight_field=None)
        return profile[field].d

    def _slice_args(self, view):
        index, coord = [(i, v) for i, v in enumerate(view)
                        if not isinstance(v, tuple)][0]
        coord = self._get_loc(coord)[index]
        return index, coord

    def _frb_args(self, view, axis):
        ix = self.ds.coordinates.x_axis[axis]
        iy = self.ds.coordinates.y_axis[axis]
        sx = view[ix]
        sy = view[iy]
        bounds = (self._dds[ix]*(sx[0]+0.5) + self._left_edge[ix],
                  self._dds[ix]*(sx[1]+0.5) + self._left_edge[ix],
                  self._dds[iy]*(sy[0]+0.5) + self._left_edge[iy],
                  self._dds[iy]*(sy[1]+0.5) + self._left_edge[iy])
        return bounds, (sy[2], sx[2])

    def compute_fixed_resolution_buffer(self, bounds, target_data=None,
                                        target_cid=None, subset_state=None,
                                        broadcast=True, cache_id=None):
        print(str(subset_state) + ' in CFRB')
        #Return data FRB slice if subset is None, else return mask of subset

        if target_cid is None and subset_state is None:
            raise ValueError("Either target_cid or subset_state should be specified")

        for bound in bounds:
            if isinstance(bound, tuple) and bound[2] < 1:
                raise ValueError("Number of steps in bounds should be >=1")

        field = cid_to_field(target_cid)
        print('Calling GSR in compute_statistic in compute_FRB')
        reg = self._get_subset_region(subset_state)
        nd = len([b for b in bounds if isinstance(b, tuple)])
        if nd == 2:
            axis, coord = self._slice_args(bounds)
            sl = self.ds.slice(axis, coord, data_source = reg)
            frb = FixedResolutionBuffer(sl, *self._frb_args(bounds, axis))
            if subset_state is None:
                return frb[field].d.T
        elif nd == 3:
            bds = np.array(bounds)
            le = self._get_loc(bds[:,0])
            re = self._get_loc(bds[:,1])
            shape = bds[:,2].astype("int")
            if np.any(le < self._left_edge) | np.any(re > self._right_edge):
                ret = np.empty(shape)
                ret[:] = np.nan
                return ret
            ag = self.ds.arbitrary_grid(le, re, shape)
            if subset_state is None:
                return ag[field].d

        try:
            att_field = sub('"','', subset_state.att.label).split(',')[1]
            cgatt = frb[att_field].d.T
            frb_mask = frb['zeros'].d.T
            #Possibly change to yt function include_inside in the future
            #Use other yt data.containers.include... fcns for other sets of logic
            wr=np.where(np.logical_and(cgatt>= subset_state.lo, cgatt<=subset_state.hi))
            frb_mask[wr[0], wr[1]] = 1
            return frb_mask
        except AttributeError:
            #Should never get here.
            print("Returning Nothing in cfrb")
            return


def is_yt_dataset(filename):
    try:
        yt.load(filename)
    except Exception:
        return False
    return True


@data_factory('yt dataset', is_yt_dataset)
def read_yt(filename):
    ds = DatasetSeries(filename)
    return YTGlueData(ds)
