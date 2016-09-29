# classes for conversion from NRML to .mat
import random
from scipy.io import savemat
from scipy.optimize import curve_fit

import numpy as np

import math

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

# from descartes import PolygonPatch

from openquake.hazardlib.source import PointSource

# import pandas as pd
import seaborn as sns
# import seaborn.apionly as sns

sns.set(style="whitegrid", color_codes=True)


class ZoneContainer:
    def __init__(self, id, polygon):
        """
        Create a ``ZoneContainer`` that can hold and organize all
        information.

        :param polygon: Edges of polygon given as (n,2) numpy array.
        :return:
        """

        # initialize source zone polygon
        self.polygon = polygon
        self.id = str(id)
        # initialize empty point sources dict, will hold references to sources
        # falling into the zone
        self.linked_point_sources = []

    def plot_min_mag_distribution(self):
        """
        Plots the distribution of minimum magnitude of all point sources belonging to this zone.

        :return:
        """
        fig = plt.figure()
        n, bins, patches = plt.hist([s.mfd.get_min_max_mag()[0] for s in self.linked_point_sources],
                                    bins=50, histtype='bar', rwidth=1)
        plt.xlabel('minimum magnitude')
        plt.ylabel('number of point sources')
        plt.title('Histogram of minimum magnitude of MFD of point sources')
        plt.grid(True)
        plt.show()

    def get_mean_latitude(self):
        """
        Get mean latitude of zone.

        :return:
        """
        return np.array(
            [p.location.latitude for p in self.linked_point_sources]).mean()

    def get_min_magnitude(self):
        """
        Get distribution of possible minimum magnitudes of zone.
        :return:
        """
        min_mags = np.array([p.mfd.get_min_max_mag()[0]
                             for p in self.linked_point_sources])
        return np.unique(min_mags, return_counts=True)

    def get_max_magnitude(self):
        """
        Get distribution of possible maximum magnitudes of zone.

        :return:
        """
        max_mags = np.array([p.mfd.get_min_max_mag()[1]
                             for p in self.linked_point_sources])
        return np.unique(max_mags, return_counts=True)

    def get_depth(self):
        """
        Get distribution of possible depths of points in zone.

        :return:
        """
        depths = np.array([p.mfd.get_min_max_mag()[1]
                             for p in self.linked_point_sources])
        return np.unique(depths, return_counts=True)

    def plot_max_mag_distribution(self):
        """
        Plots the distribution of minimum magnitude of all point sources belonging to this zone.
        :return:
        """
        fig = plt.figure()
        n, bins, patches = plt.hist([s.mfd.get_min_max_mag()[1] for s in self.linked_point_sources],
                                    bins=50, histtype='bar', rwidth=1)
        plt.xlabel('maximum magnitude')
        plt.ylabel('number of point sources')
        plt.title('Histogram of maximum magnitude of MFD of point sources')
        # plt.axis([40, 160, 0, 0.03])
        plt.grid(True)
        plt.show()

    def plot_polygon(self, basemap=None, axes=None):
        """
        Plot polygon of zone together with grid points.

        :param basemap: Basemap to plot into. Optional, otherwise create new map.
        :param axes: Axes to plot into. Optional.
        :return:
        """
        if not basemap:
            fig, ax = plt.subplots(1, 1)
            [min_lon, min_lat, max_lon, max_lat] = self.polygon.bounds
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
            m.drawcoastlines(linewidth=0.3)
            # fill color is color of ocean
            m.drawmapboundary(fill_color='#99ffff')
            m.fillcontinents(color='#cc9966', lake_color='#99ffff')
            # draw parallels and meridians and label on left and bottom of map.
            parallels = np.arange(0., 80, 1.)
            m.drawparallels(parallels, labels=[1, 0, 0, 1])
            meridians = np.arange(10., 360., 1.)
            m.drawmeridians(meridians, labels=[1, 0, 0, 1])
            # plot polygon of area zone
            polygon = Polygon(np.array(m(self.polygon.boundary.xy[0].tolist(),
                                         self.polygon.boundary.xy[1].tolist())).T,
                              fill=False, linewidth=3, color='b', zorder=10)
            # title of plot
            ax.set_title('Area Zone %s' % self.id)
        else:
            # use provided basemap
            m = basemap
            # use provided axes
            ax = axes
            # plot polygon of area zone
            polygon = Polygon(np.array(m(self.polygon.boundary.xy[0].tolist(),
                                         self.polygon.boundary.xy[1].tolist())).T,
                              fill=False, linewidth=1.5, color='k', zorder=10)
        # add polygon to axes
        ax.add_patch(polygon)


class FaultContainer:
    def __init__(self):
        """
        Create a ``FaultsContainer`` that can hold and organize all
        information.

        :return:
        """
        # initialize source zone polygon
        # initialize empty point sources dict, will hold references to faults
        # sources
