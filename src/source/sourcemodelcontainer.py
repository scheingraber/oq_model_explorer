# classes for conversion from NRML to .mat
import random
import os

import math
from scipy.optimize import curve_fit

import matplotlib as mpl
# without X server, use cairo backend
if not os.environ.has_key('DISPLAY'):
    mpl.use('cairo')

from scipy.io import savemat

import numpy as np

import scipy

from openquake.commonlib.source import SourceModelParser
from openquake.commonlib.sourceconverter import SourceConverter
from openquake.hazardlib.source import *

import shapely.geometry as shpgeo

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon

import seaborn as sns
from scipy.stats import gaussian_kde
from openquake.hazardlib.mfd.truncated_gr import TruncatedGRMFD

sns.set(style="whitegrid", color_codes=True)


def get_point_source_marker_color(point_source, point_src_color):
    """
    Get color for a point_source source marker taking different MFD into
    account.

    :param point_source: Point source object.
    :param point_src_color: The value for the point source marker color
                            colorcode. Can be 'aval, 'bval' (GR a/b-values),
                            'min_mag', 'max_mag', and 'moment_rate'
                            (seismic moment rate).
    :return:
    """
    if point_src_color.lower() == 'bval':
        # use b value of GR distribution
        return get_tr_gutenberg_richter_values(point_source)[1]
    elif point_src_color.lower() == 'aval':
        # use a value of GR distribution
        return get_tr_gutenberg_richter_values(point_source)[0]
    elif point_src_color.lower() == 'min_mag':
        return get_min_max_mag(point_source)[0]
    elif point_src_color.lower() == 'max_mag':
        return get_min_max_mag(point_source)[1]
    elif point_src_color.lower() == 'moment_rate':
        min_mag, max_mag = point_source.mfd.get_min_max_mag()
        a, b = get_tr_gutenberg_richter_values(point_source)
        return get_total_moment_rate(a, b, min_mag, max_mag)
    else:
        raise ValueError('Unsupported color code string.')


def get_min_max_mag(source):
    """
    Get minimum and maximum magnitude of source, preferably from fields of
    ``source``, otherwise from method ``get_min_max_mag`` of source.

    :param source: Openquake source object.
    :return:
    """
    if hasattr(source.mfd, 'min_mag'):
        min_mag = source.mfd.min_mag
    else:
        min_mag = source.mfd.get_min_max_mag()[0]
    if hasattr(source.mfd, 'max_mag'):
        max_mag = source.mfd.max_mag
    else:
        max_mag = source.mfd.get_min_max_mag()[1]
    return [min_mag, max_mag]


def get_point_source_marker_size(point_source):
    """
    Get sizes for point source markers taking different MFD into account.

    :param point_source: Point source object.
    :return
    """
    # get Gutenberg-Richter fit
    bval = get_tr_gutenberg_richter_values(point_source)[1]
    # or get moment rate
    return 7 ** (bval + 1)


def get_total_moment_rate(a, b, min_mag, max_mag):
    """
    Calculate total moment rate of ``point_source``(total energy released per
    unit time) given a truncated Gutenberg-Richter distribution with
    a, b, min_mag, max_mag ::

        TMR = ((10**ai) / bi) * (10 ** (bi*max_mag) - 10 ** (bi*min_mag))

    where ``ai = a + log10(b) + 9.05`` and ``bi = 1.5 - b``.
    In case of ``bi == 0`` the following formula is applied::

        TMR = (10 ** ai) * (max_mag - min_mag)

    :param a: Parameter ``a`` of truncated Gutenberg-Richter distribution.
    :param b: Parameter ``b`` of truncated Gutenberg-Richter distribution.
    :param min_mag: Minimum magnitude.
    :param max_mag: Maximum magnitude.
    :return
        Float, calculated TMR value in ``N * m / year``
        (Newton-meter per year).
    """
    ai = 9.05 + a + math.log10(b)
    bi = 1.5 - b
    if bi == 0.0:
        return (10 ** ai) * (max_mag - min_mag)
    else:
        return (((10 ** ai) / bi) *
                (10 ** (bi * max_mag) - 10 ** (bi * min_mag)))


def get_tr_gutenberg_richter_values(source):
    """
    Return the parameters of the truncated Gutenberg-Richter MFD of
    ``source``.

    The annual occurrence rate for a specific bin (magnitude band)
    is defined as ::

        rate = 10 ** (a_val - b_val * mag_lo) - 10 ** (a_val - b_val * mag_hi)

    where

    * ``a_val`` is the cumulative ``a`` value (``10 ** a`` is the number
      of exponential distribution. It describes the relative size distribution
      of earthquakes: a higher ``b`` value indicates a relatively larger
      proportion of small events and vice versa.
    * ``mag_lo`` and ``mag_hi`` are lower and upper magnitudes of a specific
      bin respectively.

    :param source: Source object.
    :return:
        Dictionary of parameters a,b,min_mag,max_mag of the truncated
        Gutenberg-Richter distribution.
    """
    if isinstance(source.mfd, TruncatedGRMFD):
        # just return GR values
        return source.mfd.a_val, source.mfd.b_val
    else:
        # fit truncated GR distribution
        rates = source.mfd.get_annual_occurrence_rates()
        rates_nonacc = np.array([r[1] for r in rates])
        rates_acc = np.cumsum(rates_nonacc[::-1])[::-1]
        mags = np.array([r[0] for r in rates])
        # obtain optimal parameters and their covariance
        try:
            popt, pcov = curve_fit(get_tr_gutenberg_richter_occ_rates, mags,
                                   rates_nonacc)
        except:
            popt = [np.nan, np.nan]
        return popt


def get_tr_gutenberg_richter_occ_rates(mags, a_val, b_val):
    """
    Calculate and return an annual occurrence rate for a specific magnitude bin.

    :param mags: Dict describing minimum and maximum of mag bin.
    :param a_val: a-value of GR distribution.
    :param b_val: b-value of GR distribution.
    :returns:
        Float number, the annual occurrence rate calculated using formula
        described in :class:`TruncatedGRMFD`.
    """
    # obtain bin width (constant)
    if len(mags) > 1:
        bin_width = mags[1] - mags[0]
    else:
        bin_width = np.nan
    occ_rates = []
    for i in range(len(mags)):
        occ_rates.append(10 ** (a_val - b_val * (mags[i] - bin_width / 2.0)) -
                         10 ** (a_val - b_val * (mags[i] + bin_width / 2.0)))
    return np.array(occ_rates)


def get_planar_surface_boundary(surf):
    """
    Return coordinates of planar surface boundary

    :param surf: Surface of source.
    """
    boundary_lons = np.array([surf.top_left.longitude,
                              surf.top_right.longitude,
                              surf.bottom_right.longitude,
                              surf.bottom_left.longitude,
                              surf.top_left.longitude])
    boundary_lats = np.array([surf.top_left.latitude,
                              surf.top_right.latitude,
                              surf.bottom_right.latitude,
                              surf.bottom_left.latitude,
                              surf.top_left.latitude])
    return boundary_lons, boundary_lats


def get_mesh_boundary(mesh):
    """
    Return coordinates of mesh boundary

    :param mesh: Mesh of fault.
    """
    boundary_lons = np.concatenate((mesh.lons[0, :],
                                    mesh.lons[1:, -1],
                                    mesh.lons[-1, :-1][::-1],
                                    mesh.lons[:-1, 0][::-1]))
    boundary_lats = np.concatenate((mesh.lats[0, :],
                                    mesh.lats[1:, -1],
                                    mesh.lats[-1, :-1][::-1],
                                    mesh.lats[:-1, 0][::-1]))

    return boundary_lons, boundary_lats


def get_map_projection(sources, topo_map=False):
    """
    Return map projection specific to sources.

    :param topo_map: Bool, whether to plot topographical map.
    :param sources: Openquake Sources as list.
    :return fig: Figure handle.
            ax: Axes handle.
            m: Basemap handle.
    """
    # boundary from point sources
    if len(sources['point']) > 0:
        min_lon = min(p.location.longitude for p in sources['point'])
        max_lon = max(p.location.longitude for p in sources['point'])
        min_lat = min(p.location.latitude for p in sources['point'])
        max_lat = max(p.location.latitude for p in sources['point'])
    else:
        min_lon = 180
        max_lon = -180
        min_lat = 90
        max_lat = -90
    # boundary from complex sources
    for src in sources['complex']:
        # extract rupture enclosing polygon (considering a buffer of 20 km)
        rup_poly = src.get_rupture_enclosing_polygon(20.)
        min_lon = np.min(np.hstack([rup_poly.lons, min_lon]))
        max_lon = np.max(np.hstack([rup_poly.lons, max_lon]))
        min_lat = np.min(np.hstack([rup_poly.lats, min_lat]))
        max_lat = np.max(np.hstack([rup_poly.lats, max_lat]))
    # create new figure and axes
    fig, ax = plt.subplots(1, 1)
    # create new map
    m = Basemap(
        projection='merc',
        ellps='WGS84',
        llcrnrlon=min_lon - 0.2,
        llcrnrlat=min_lat - 0.2,
        urcrnrlon=max_lon + 0.2,
        urcrnrlat=max_lat + 0.2,
        lat_ts=0,
        lon_0=(min_lon + max_lon) / 2.0,
        resolution='h')
    if topo_map:
        # topographical map
        m.etopo()
        # add country borders and map boundary
        m.drawcountries()
        m.drawmapboundary()
    else:
        # normal country map
        m.drawcoastlines(linewidth=0.3)
        # fill color is color of ocean
        m.drawmapboundary(fill_color='#99ffff')
        m.fillcontinents(color='#cc9966', lake_color='#99ffff')
    # draw parallels and meridians and label on left and bottom of map.
    # draw 6 lines and round to integer
    parallels = np.rint(np.linspace(min_lat, max_lat, 6))
    m.drawparallels(parallels, labels=[1, 0, 0, 1])
    meridians = np.rint(np.linspace(min_lon, max_lon, 6))
    m.drawmeridians(meridians, labels=[1, 0, 0, 1])
    return fig, ax, m


def compile_point_sources(sources, split_depth=None,
                          min_depth=None, max_depth=None):
    """
    Compile all raw point sources information to save to .mat.

    :param sources: List of point sources.
    :param split_depth: Optional, if splitting into shallow and deep point
                        sources is desired.
    :param min_depth: Optional, minimum depth of a point source.
    :param max_depth: Optional, maximum depth of a point source.
    :return: Dictionary of point source information for ``scipy.io.savemat``.
    """
    points = dict()
    if split_depth:
        # split into shallow and deep seismicity
        points['shallow'] = dict()
        points['deep'] = dict()
        for key in ['shallow', 'deep']:
            points[key]['name'] = []
            points[key]['tectonic_region'] = []
            points[key]['location'] = np.zeros([len(sources), 2])
            points[key]['seismo_depth_range'] = np.zeros([len(sources), 2])
            points[key]['mag_scale'] = []
            points[key]['rup_aspect_ratio'] = np.zeros([len(sources), 1])
            points[key]['rates'] = []
            points[key]['gr_params'] = []
            points[key]['mag_range'] = []
            points[key]['nodalplane_dist'] = []
            points[key]['hypodepth_dist'] = []
        i = 0
        for p in sources:
            # skip to next point if below min depth or above max depth
            if min_depth and min_depth > p.hypocenter_distribution.data[0][1]:
                continue
            if max_depth and max_depth < p.hypocenter_distribution.data[0][1]:
                continue
            if p.hypocenter_distribution.data[0][1] < split_depth:
                key = 'shallow'
            else:
                key = 'deep'
            points[key]['name'].append(p.name)
            points[key]['tectonic_region'].append(p.tectonic_region_type)
            points[key]['location'][i, 0] = p.location.longitude
            points[key]['location'][i, 1] = p.location.latitude
            points[key]['seismo_depth_range'][i, 0] = p.lower_seismogenic_depth
            points[key]['seismo_depth_range'][i, 1] = p.upper_seismogenic_depth
            points[key]['mag_scale'].append(repr(p.magnitude_scaling_relationship))
            points[key]['rup_aspect_ratio'][i, 0] = p.rupture_aspect_ratio
            # get rates
            points[key]['rates'].append(np.array(p.mfd.get_annual_occurrence_rates()))
            # fit truncated Gutenberg-Richter distribution
            points[key]['gr_params'].append(get_tr_gutenberg_richter_values(p))
            points[key]['mag_range'].append(get_min_max_mag(p))
            # nodal plane distribution
            npdist = dict()
            npdist['prob'] = [nplane[0] for nplane in p.nodal_plane_distribution.data]
            npdist['strike'] = [nplane[1].strike for nplane in p.nodal_plane_distribution.data]
            npdist['dip'] = [nplane[1].dip for nplane in p.nodal_plane_distribution.data]
            npdist['rake'] = [nplane[1].rake for nplane in p.nodal_plane_distribution.data]
            points[key]['nodalplane_dist'].append(npdist)
            # TODO: fix this for the case of multiple hypodepths
            points[key]['hypodepth_dist'].append(
                np.array(p.hypocenter_distribution.data).squeeze())
            i += 1
    else:
        # do not split into shallow and deep seismicity points
        points['name'] = []
        points['tectonic_region'] = []
        points['location'] = np.zeros([len(sources), 2])
        points['seismo_depth_range'] = np.zeros([len(sources), 2])
        points['mag_scale'] = []
        points['rup_aspect_ratio'] = np.zeros([len(sources), 1])
        points['rates'] = []
        points['gr_params'] = []
        points['mag_range'] = []
        points['nodalplane_dist'] = []
        points['hypodepth_dist'] = []
        i = 0
        for p in sources:
            if max_depth and max_depth < p.hypocenter_distribution.data[0][1]:
                # skip to next point
                continue
            points['name'].append(p.name)
            points['tectonic_region'].append(p.tectonic_region_type)
            points['location'][i, 0] = p.location.longitude
            points['location'][i, 1] = p.location.latitude
            points['seismo_depth_range'][i, 0] = p.lower_seismogenic_depth
            points['seismo_depth_range'][i, 1] = p.upper_seismogenic_depth
            points['mag_scale'].append(repr(p.magnitude_scaling_relationship))
            points['rup_aspect_ratio'][i, 0] = p.rupture_aspect_ratio
            # get rates
            points['rates'].append(np.array(p.mfd.get_annual_occurrence_rates()))
            # fit truncated Gutenberg-Richter distribution
            points['gr_params'].append(get_tr_gutenberg_richter_values(p))
            points['mag_range'].append(get_min_max_mag(p))
            # nodal plane distribution
            npdist = dict()
            npdist['prob'] = [nplane[0] for nplane in p.nodal_plane_distribution.data]
            npdist['strike'] = [nplane[1].strike for nplane in p.nodal_plane_distribution.data]
            npdist['dip'] = [nplane[1].dip for nplane in p.nodal_plane_distribution.data]
            npdist['rake'] = [nplane[1].rake for nplane in p.nodal_plane_distribution.data]
            points['nodalplane_dist'].append(npdist)
            # TODO: check if this works for the case of multiple hypodepths
            points['hypodepth_dist'].append(
                np.array(p.hypocenter_distribution.data).squeeze())
            i += 1
    return points


def compile_simple_fault_sources(sources):
    """
    Compile all raw simple fault sources information to save to .mat.

    :param sources: List of simple fault sources.
    :return: Dictionary of simple fault source information for ``scipy.io.savemat``.
    """
    faults = dict()
    faults['name'] = []
    faults['tectonic_region'] = []
    faults['position_list'] = []
    faults['dip'] = np.zeros([len(sources), 1])
    faults['rake'] = np.zeros([len(sources), 1])
    faults['seismo_depth_range'] = np.zeros([len(sources), 2])
    faults['mag_scale'] = []
    faults['gr_params'] = []
    faults['rup_aspect_ratio'] = np.zeros([len(sources), 1])
    faults['rates'] = []
    for i, s in enumerate(sources):
        faults['name'].append(s.name)
        faults['tectonic_region'].append(s.tectonic_region_type)
        pos_list = np.zeros([len(s.fault_trace.points), 2])
        pos_list[:, 0] = np.array(
            [pt.longitude for pt in s.fault_trace.points])
        pos_list[:, 1] = np.array([pt.latitude for pt in s.fault_trace.points])
        faults['position_list'].append(pos_list)
        faults['gr_params'].append(get_tr_gutenberg_richter_values(s))
        faults['dip'][i] = s.dip
        faults['rake'][i] = s.rake
        faults['seismo_depth_range'][i, 0] = s.lower_seismogenic_depth
        faults['seismo_depth_range'][i, 1] = s.upper_seismogenic_depth
        faults['mag_scale'].append(repr(s.magnitude_scaling_relationship))
        faults['rup_aspect_ratio'][i, 0] = s.rupture_aspect_ratio
        # get rates
        faults['rates'].append(np.array(s.mfd.get_annual_occurrence_rates()))
    return faults


def compile_area_sources(sources):
    """
    Compile all area sources information to save to .mat.

    :param sources: List of complex fault sources.
    :return: Dictionary of area source information for ``scipy.io.savemat``.
    """
    area = dict()
    area['name'] = []
    area['tectonic_region'] = []
    area['polygon'] = []
    area['discretization'] = np.zeros([len(sources), 1])
    area['seismo_depth_range'] = np.zeros([len(sources), 2])
    area['mag_scale'] = []
    area['rup_aspect_ratio'] = np.zeros([len(sources), 1])
    area['rates'] = []
    area['gr_params'] = []
    area['mag_range'] = []
    area['nodalplane_dist'] = []
    area['hypodepth_dist'] = []
    i = 0
    for s in sources:
        area['name'].append(s.name)
        area['tectonic_region'].append(s.tectonic_region_type)
        poly = np.zeros([len(s.polygon.lons), 2])
        poly[:, 0] = s.polygon.lons
        poly[:, 1] = s.polygon.lats
        area['polygon'].append(poly)
        area['seismo_depth_range'][i, 0] = s.lower_seismogenic_depth
        area['seismo_depth_range'][i, 1] = s.upper_seismogenic_depth
        area['mag_scale'].append(repr(s.magnitude_scaling_relationship))
        area['rup_aspect_ratio'][i, 0] = s.rupture_aspect_ratio
        # get rates
        area['rates'].append(np.array(s.mfd.get_annual_occurrence_rates()))
        # fit truncated Gutenberg-Richter distribution
        area['gr_params'].append(get_tr_gutenberg_richter_values(s))
        area['mag_range'].append(get_min_max_mag(s))
        # nodal plane distribution
        npdist = dict()
        npdist['prob'] = [nplane[0] for nplane in s.nodal_plane_distribution.data]
        npdist['strike'] = [nplane[1].strike for nplane in s.nodal_plane_distribution.data]
        npdist['dip'] = [nplane[1].dip for nplane in s.nodal_plane_distribution.data]
        npdist['rake'] = [nplane[1].rake for nplane in s.nodal_plane_distribution.data]
        area['nodalplane_dist'].append(npdist)
        # TODO: fix this for the case of multiple hypodepths
        area['hypodepth_dist'].append(
            np.array(s.hypocenter_distribution.data).squeeze())
        i += 1
    return area


def compile_complex_fault_sources(sources):
    """
    Compile all raw complex fault sources information to save to .mat.

    :param sources: List of complex fault sources.
    :return: Dictionary of complex fault source information for ``scipy.io.savemat``.
    """
    faults = dict()
    faults['name'] = []
    faults['tectonic_region'] = []
    faults['edges'] = []
    faults['gr_params'] = []
    faults['rake'] = np.zeros([len(sources), 1])
    faults['mag_scale'] = []
    faults['rup_aspect_ratio'] = np.zeros([len(sources), 1])
    faults['rates'] = []
    for i, s in enumerate(sources):
        faults['name'].append(s.name)
        faults['tectonic_region'].append(s.tectonic_region_type)
        faults['edges'].append([])
        for e in s.edges:
            pos_list = np.zeros([len(e.points), 3])
            pos_list[:, 0] = np.array([pt.longitude for pt in e.points])
            pos_list[:, 1] = np.array([pt.latitude for pt in e.points])
            pos_list[:, 2] = np.array([pt.depth for pt in e.points])
            faults['edges'][i].append(pos_list)
        faults['rake'][i] = s.rake
        faults['mag_scale'].append(repr(s.magnitude_scaling_relationship))
        faults['rup_aspect_ratio'][i, 0] = s.rupture_aspect_ratio
        # get rates
        faults['rates'].append(np.array(s.mfd.get_annual_occurrence_rates()))
        faults['gr_params'].append(get_tr_gutenberg_richter_values(s))
    return faults


def compile_char_fault_sources(sources):
    """
    Compile all raw characteristic fault sources information to save to .mat.

    :param sources: List of characteristic fault sources.
    :return: Dictionary of characteristic fault source information for ``scipy.io.savemat``.
    """
    faults = dict()
    faults['name'] = []
    faults['tectonic_region'] = []
    faults['mesh'] = []
    faults['rake'] = np.zeros([len(sources), 1])
    faults['mag_scale'] = []
    faults['rup_aspect_ratio'] = np.zeros([len(sources), 1])
    faults['rates'] = []
    for i, s in enumerate(sources):
        faults['name'].append(s.name)
        faults['tectonic_region'].append(s.tectonic_region_type)
        mesh = dict()
        mesh['lon'] = s.surface.mesh.lons
        mesh['lat'] = s.surface.mesh.lats
        mesh['depth'] = s.surface.mesh.depths
        faults['mesh'].append(mesh)
        faults['rake'][i] = s.rake
        faults['mag_scale'].append(repr(s.magnitude_scaling_relationship))
        faults['rup_aspect_ratio'][i, 0] = s.rupture_aspect_ratio
        # get rates
        faults['rates'].append(np.array(s.mfd.get_annual_occurrence_rates()))
    return faults


class NrmlModelContainer:
    """
    This class contains a NRML source model and provides plotting
    functionality.
    """

    def __init__(self, nrml_filename, min_depth=None, max_depth=None):
        """
        Create a container for an NRML source model.

        :param nrml_filename:
            Filename of input NRML source model (string).
        :param min_depth: Optional, minimum depth of a point source.
        :param max_depth: Optional, maximum depth of a point source.
        """
        self.max_depth = max_depth
        self.min_depth = min_depth
        # instantiate Source converter
        converter = SourceConverter(investigation_time=50,
                                    rupture_mesh_spacing=5.0,
                                    complex_fault_mesh_spacing=20.0,
                                    width_of_mfd_bin=1.0,
                                    area_source_discretization=20.0)
        src_parser = SourceModelParser(converter)
        # parse sources from xml file
        self.sources_orig = src_parser.parse_sources(nrml_filename)
        # initialize lists holding source types
        self.sources = {'point': [], 'area': [], 'simple': [], 'complex': [],
                        'characteristic': [], 'nonparametric': []}
        # populate self.sources dict
        self.collect_source_containers()

    def collect_source_containers(self):
        """
        Collect all sources in dictionary according to their type.

        :return:
        """
        # loop over sources
        for s in self.sources_orig:
            if type(s) is PointSource:
                # skip to next point if below min depth or above max depth
                if self.min_depth and self.min_depth > s.hypocenter_distribution.data[0][1]:
                    continue
                if self.max_depth and self.max_depth < s.hypocenter_distribution.data[0][1]:
                    continue
                self.sources['point'].append(s)
            elif type(s) is SimpleFaultSource:
                self.sources['simple'].append(s)
            elif type(s) is CharacteristicFaultSource:
                self.sources['characteristic'].append(s)
            elif type(s) is AreaSource:
                self.sources['area'].append(s)
            elif type(s) is ComplexFaultSource:
                self.sources['complex'].append(s)
            elif type(s) is NonParametricSeismicSource:
                self.sources['nonparametric'].append(s)
            else:
                raise "Unknown Source type: %s" % type(s)

    def save_unconverted_mat(self, filename, min_depth=None, max_depth=None):
        """
        Save to .mat file containing all unconverted and unreduced information.

        :param filename: File to write.
        :param min_depth: Optional, minimum depth of a point source.
        :param max_depth: Optional, maximum depth of a point source.
        :return:
        """
        mat_dict = dict()
        # compile raw source information
        if len(self.sources['point']) > 0:
            mat_dict['pointSources'] = compile_point_sources(
                                            sources=self.sources['point'],
                                            min_depth=min_depth,
                                            max_depth=max_depth)
        if len(self.sources['simple']) > 0:
            mat_dict[
                'simpleFaultSources'] = compile_simple_fault_sources(
                                            self.sources['simple'])
        if len(self.sources['complex']) > 0:
            mat_dict[
                'complexFaultSources'] = compile_complex_fault_sources(
                                            self.sources['complex'])
        if len(self.sources['characteristic']) > 0:
            mat_dict['charFaultSources'] = compile_char_fault_sources(
                                            self.sources['characteristic'])
        if len(self.sources['area']) > 0:
            mat_dict['areaSources'] = compile_area_sources(
                                            self.sources['area'])
        # TODO: implement saving for nonparametric sources
        assert len(self.sources['nonparametric']) == 0
        savemat(filename, mat_dict, oned_as='column')

    def plot(self, basemap=None, axes=None, max_point_sources=None,
             topo_map=False, filename=None, dpi=600, density_hack=False,
             point_src_color='bval', point_src_size=80):
        """
        Plot a representation of the Nrml Model Container into basemap and axes.

        :param topo_map: Bool, whether to plot topographical map.
        :param basemap: Basemap to plot into.
        :param axes: Axes to plot into.
        :param max_point_sources: Maximum number of point sources to plot.
        :param filename: Filename to save to. If None, plot to figure window.
        :param dpi: DPI if saving to raster file.
        :param density_hack: Fit a gaussian kde and plot densest points last.
                             This looks better for many overlapping points.
        :param point_src_color: The value for the point source marker color
                                colorcode. Can be 'bval' (GR b-value, default),
                                'min_mag', 'max_mag', and 'moment_rate'
                                (seismic moment rate).
        :param point_src_size: Size of point sources for scatter plot given as
                               integer, or 'bval'.
        :return:
        """
        if not basemap:
            # get new map projection
            fig, ax, m = get_map_projection(self.sources, topo_map)
        else:
            # use provided basemap
            m = basemap
            # use provided axes
            ax = axes
        # plot point sources
        if len(self.sources['point']) > 0:
            # plot point sources (only max_point_sources or all)
            if max_point_sources and \
                            max_point_sources <= len(self.sources['point']):
                # pick random point sources to plot
                rand_idx = random.sample(range(len(self.sources['point'])),
                                         max_point_sources)
                # coordinate transformation to basemap
                x, y = m([self.sources['point'][i].location.longitude for i in
                          rand_idx],
                         [self.sources['point'][i].location.latitude for i in
                          rand_idx])
                # marker sizes and colors
                s = [get_point_source_marker_size(self.sources['point'][i])
                     for i in rand_idx]
                c = [get_point_source_marker_color(self.sources['point'][i],
                                                   point_src_color)
                     for i in rand_idx]
            else:
                # plot all
                # coordinate transformation to basemap
                x, y = m([p.location.longitude for p in self.sources['point']],
                         [p.location.latitude for p in self.sources['point']])
                # marker sizes and colors
                if point_src_size == 'bval':
                    s = [get_point_source_marker_size(p) for p in
                         self.sources['point']]
                else:
                    # just use provided integer
                    s = point_src_size
                # get point source color
                c = [get_point_source_marker_color(p, point_src_color) for p in
                     self.sources['point']]
            # if density hack enabled, sort points by density to produce nicer
            # scatter plot
            if density_hack:
                x = np.array(x)
                y = np.array(y)
                # calculate the point density
                xy = np.vstack([x, y])
                z = gaussian_kde(xy)(xy)
                # Sort the points by density, densest points are plotted last
                idx = z.argsort()
                x, y, z = x[idx], y[idx], z[idx]
            # plot point sources into map and add colorbar to the right
            sh = ax.scatter(x, y, c=c, s=s, marker='o', cmap='jet', alpha=0.4,
                            edgecolor='k', picker=True, zorder=2)
            cb = m.colorbar(sh, location='right')
            cb.set_label(point_src_color)
        # plot simple faults
        if len(self.sources['simple']) > 0:
            for src in self.sources['simple']:
                # plot simple fault
                lons = [p.longitude for p in src.fault_trace.points]
                lats = [p.latitude for p in src.fault_trace.points]
                x, y = m(lons, lats)
                m.plot(x, y, linewidth=2, color='red', zorder=5)
        # plot complex faults
        if len(self.sources['complex']) > 0:
            for src in self.sources['complex']:
                # get surface projection
                p = src.get_rupture_enclosing_polygon()
                verts = np.array([p.lons, p.lats])
                poly_patch = Polygon(np.array(m(verts[0, :], verts[1, :])).T,
                                     facecolor='red', alpha=0.4,
                                     edgecolor='black', linewidth=3, zorder=4)
                ax.add_patch(poly_patch)
        # plot characteristic faults
        if len(self.sources['characteristic']) > 0:
            boundaries = []
            for src in self.sources['characteristic']:
                # get surface projection
                lons, lats = get_mesh_boundary(src.surface.mesh)
                xx, yy = m(lons, lats)
                boundaries.append([(x, y) for x, y in zip(xx, yy)])
            boundaries = np.array(boundaries)
            patch_collection = PolyCollection(boundaries,
                                              facecolors='palegreen',
                                              zorder=5)
            ax.add_collection(patch_collection)
        # plot area sources
        if len(self.sources['area']) > 0:
            boundaries = []
            for src in self.sources['area']:
                # get surface projection boundary
                xx, yy = m(src.polygon.lons, src.polygon.lats)
                boundaries.append([(x, y) for x, y in zip(xx, yy)])
            boundaries = np.array(boundaries)
            patch_collection = PolyCollection(boundaries, alpha=0.4,
                                              facecolors='palegreen',
                                              zorder=3)
            ax.add_collection(patch_collection)

    def plot_source_mfd(self, idx):
        """
        Plot magnitude frequency distribution of source given by index ``idx``.

        :param idx: Index of source.
        :return:
        """
        # get point source from ``nrml_model_container``.
        p = self.sources['point'][idx]
        f, (ax1, ax2) = plt.subplots(1, 2)
        # get annual occurence rates of different magnitudes
        rates = p.mfd.get_annual_occurrence_rates()
        rates_nonacc = [r[1] for r in rates]
        rates_acc = np.cumsum(rates_nonacc[::-1])[::-1]
        mags = [r[0] for r in rates]
        # accumulate s.t. each rate is the exceedance rate
        ax1.set_xlabel('magnitude')
        ax1.set_ylabel('log(annual occurrence rate)')
        ax1.plot(mags, np.log(rates_nonacc), linewidth='4.0',
                 label='Original incremental MFD')
        ax2.set_xlabel('magnitude')
        ax2.set_ylabel('log(annual exceedance rate)')
        ax2.plot(mags, np.log(rates_acc), linewidth='4.0',
                 label='Original incremental MFD')
        # fit GR distribution and plot in dashed red
        gr_params = get_tr_gutenberg_richter_values(p)
        rates_fit = get_tr_gutenberg_richter_occ_rates(mags,
                                                       gr_params[0],
                                                       gr_params[1])
        rates_fit_acc = np.cumsum(rates_fit[::-1])[::-1]
        ax1.plot(mags, np.log(rates_fit), 'r', linestyle='--',
                 label='Fitted trunc. GR distribution (a=%f, b=%f)' %
                       (gr_params[0], gr_params[1]))
        ax2.plot(mags, np.log(rates_fit_acc), 'r', linestyle='--',
                 label='Fitted trunc. GR distribution (a=%f, b=%f)' %
                       (gr_params[0], gr_params[1]))
        # show all legends
        ax1.legend()
        ax2.legend()
        # open new window
        f.show()
