#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, \
    NavigationToolbar2TkAgg
import shapely.geometry as shpgeo

if sys.version_info.major == 2:
    # We are using Python 2.x
    import Tkinter as tk
elif sys.version_info.major == 3:
    # We are using Python 3.x
    import tkinter as tk

import tkFileDialog
import tkMessageBox
import tkSimpleDialog

from openquake.commonlib.node import read_nodes, LiteralNode
from openquake.commonlib import nrml

import filter_source_model_adapted as filter_source
from src.source.sourcemodelcontainer import NrmlModelContainer, ModelContainer
from src.source.sourceconverter import SourceModelConverter


class MapModel:
    """
    A MapModel holds the main map and polygon selection attributes.
    """

    def __init__(self):
        """
        Map Model will initialize sources, selection polygon, valid sources.

        :return:
        """
        # will hold sources objects
        self.sources = None
        # will hold selection polygon
        self.polygon = np.array([]).reshape(0, 2)
        # will hold sources that are inside currently selected polygon
        self.valid_sources = None
        # flag to store if polygon selection is finished
        self.is_finished = False
        # maximum distance of sources to polygon for filtering in km
        self.distance = 200
        # will hold grid discretization of polygon
        self.grid = None

    def reset_polygon(self):
        """
        Will reset selection polygon.
        :return:
        """
        self.polygon = np.array([]).reshape(0, 2)


class StatusBar(tk.Frame):
    """
    Status Bar for Map View.
    """

    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.label = tk.Label(self, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.label.pack(fill=tk.X)
        self.pack(side=tk.BOTTOM, fill=tk.X)

    def set(self, str_message, *args):
        """
        Set status bar to str message
        :param str_message: Message to print to status bar.
        :param args:
        :return:
        """
        self.label.config(text=str_message % args)
        self.label.update_idletasks()

    def clear(self):
        """
        Clear status bar.
        :return:
        """
        self.label.config(text="")
        self.label.update_idletasks()


class MapView:
    """
    The MapView class contains all methods to display the map model in 2D.
    """
    def __init__(self, master):
        """
        Initializes view Object.

        :param master:
        """
        # hold lines of polygon (to remove when polygon is reset)
        self.polylines = []
        self.points = []
        self.poly_patch = None
        # Initialize status bar
        self.statusbar = StatusBar(master)

        # instantiate GUI top_panel.
        self.top_frame = tk.Frame(master)
        self.top_panel = TopPanel(self.top_frame)
        self.top_frame.pack(side=tk.TOP)
        self.top_frame2 = tk.Frame(master)
        self.top_frame2.pack(side=tk.TOP)
        self.frame = tk.Frame(master)
        self.fig = plt.figure(figsize=(40, 25), dpi=100)
        self.axes = self.fig.add_axes([0.05, 0.05, 0.9, 0.9])
        self.frame.pack(side=tk.BOTTOM)
        # basemap resolution=None ok and faster if not using drawcountries
        self.map = Basemap(llcrnrlon=-180.0,
                           llcrnrlat=-90.0,
                           urcrnrlon=180.0,
                           urcrnrlat=90.0,
                           resolution=None,
                           # resolution='l',
                           projection='robin',
                           lat_0=0., lon_0=0., lat_ts=0.)
        self.map.etopo()
        # add country borders and map boundary
        # self.map.drawcountries()
        self.map.drawmapboundary()
        # draw parallels and meridians and label on left and bottom of map.
        parallels = np.arange(-90., 90, 30.)
        self.map.drawparallels(parallels, labels=[1, 0, 0, 1])
        meridians = np.arange(0., 360., 30.)
        self.map.drawmeridians(meridians, labels=[1, 0, 0, 1])
        # add map figure canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # add canvas toolbar
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.top_frame2)
        self.toolbar.update()
        # show canvas
        self.canvas.show()


class TopPanel:
    """
    Top Panel.
    """
    def __init__(self, root):
        # Add buttons to view
        # TODO: change buttons to menu layout
        self.frame2 = tk.Frame(root)
        self.frame2.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.openNrmlButton = tk.Button(master=self.frame2,
                                        text="Open NRML Source Model")
        self.openNrmlButton.pack(side=tk.LEFT)

        self.selectPolygonButton = tk.Button(master=self.frame2,
                                             text="Start Region Selection")
        self.selectPolygonButton.pack(side=tk.LEFT)

        self.finishPolygonButton = tk.Button(master=self.frame2,
                                             text="Finish Region Selection")
        self.finishPolygonButton.pack(side=tk.LEFT)

        self.resetPolygonButton = tk.Button(master=self.frame2,
                                            text="Reset Region Selection")
        self.resetPolygonButton.pack(side=tk.LEFT)

        self.analyzeZoneButton = tk.Button(master=self.frame2,
                                                 text="Analyze Zone")
        self.analyzeZoneButton.pack(side=tk.LEFT)

        self.saveButton = tk.Button(master=self.frame2,
                                    text="Save Source Model")
        self.saveButton.pack(side=tk.LEFT)

        self.discretizePolygonButton = tk.Button(master=self.frame2,
                                                 text="Discretize Region")
        self.discretizePolygonButton.pack(side=tk.LEFT)

        self.exitButton = tk.Button(master=self.frame2, text="Exit")
        self.exitButton.pack(side=tk.LEFT)


class UiController:
    def __init__(self):
        # Initializes the GUI root
        self.root = tk.Tk()

        # Initialize map model
        self.map_model = MapModel()
        # Will hold converter and models when loaded
        self.source_model_converter = None
        self.src_model_container = None
        self.nrml_model_container = None

        # Initialize map view
        self.map_view = MapView(self.root)

        # connect button press event to handlers
        self.map_view.top_panel.openNrmlButton.bind(
                                    "<Button>", self.open_nrml_source_model)
        self.map_view.top_panel.selectPolygonButton.bind(
                                    "<Button>", self.select_polygon)
        self.map_view.top_panel.finishPolygonButton.bind(
                                    "<Button>", self.finish_polygon)
        self.map_view.top_panel.discretizePolygonButton.bind(
                                    "<Button>", self.discretize_polygon)
        self.map_view.top_panel.resetPolygonButton.bind(
                                    "<Button>", self.reset_polygon)
        self.map_view.top_panel.analyzeZoneButton.bind(
                                    "<Button>", self.analyze_zone)
        self.map_view.top_panel.saveButton.bind(
                                    "<Button>", self.save_source_model)
        self.map_view.top_panel.exitButton.bind("<Button>", sys.exit)

        self.root.protocol("WM_DELETE_WINDOW", self.ask_quit)

        # set status bar to idle
        self.map_view.statusbar.set("Idle. No hazard model loaded.")
        # these will hold connection ids of mouse button events
        self.psel_cid = None
        self.sc_cid = None
        # will hold filenames of loaded models
        self.nrml_infile = None
        self.src_infile = None
        # connect artist pick in map_view to event handler
        self.sc_cid = self.map_view.canvas.callbacks.connect(
                                    'pick_event', self.analyze_source_handler)

    def ask_quit(self):
        if tkMessageBox.askokcancel("Quit", "You want to quit now?"):
            self.root.destroy()
            sys.exit(0)

    def run(self):
        """
        Run Controller Main Event Loop.
        """
        self.root.title("Openquake Source Model Extraction Tool")
        self.root.deiconify()
        self.root.mainloop()

    def open_nrml_source_model(self, event):
        """
        Open a source model and create an generator for the XML.

        :param event: Tkinter event.
        """
        # graphical file selection
        self.nrml_infile = tkFileDialog.askopenfile()
        if self.nrml_infile:
            # create a new NrmlModelContainer
            self.nrml_model_container = NrmlModelContainer(self.nrml_infile)
            # update statusbar
            self.map_view.statusbar.set("Loaded hazard model " +
                                        self.nrml_infile.name + ".")
            # plot nrml model
            sh = self.nrml_model_container.plot(basemap=self.map_view.map,
                                                axes=self.map_view.axes,
                                                max_point_sources=500)
            self.map_view.canvas.draw()
        # create source model converter if both models loaded
        if self.nrml_model_container and self.src_model_container:
            self.source_model_converter = SourceModelConverter(
                                            self.nrml_model_container,
                                            self.src_model_container)

    def save_source_model(self, event):
        """
        Choose xml filename and save filtered sources.

        :param event: Tkinter event.
        """
        # save file selection dialog
        nrml_outfile = tkFileDialog.asksaveasfilename()
        if nrml_outfile:
            output_name = {"name": nrml_outfile}
            output_source_model = LiteralNode(
                                        "sourceModel", output_name,
                                        nodes=self.map_model.valid_sources)
            with open(nrml_outfile, "w") as f:
                nrml.write([output_source_model], f, "%s")
                self.map_view.statusbar.set("Saved filtered model to %s."
                                            % nrml_outfile)

    def save_grid(self, event):
        """
        Choose csv filename and save grid discretization.

        :param event: Tkinter event.
        :param self:
        """
        # save file selection dialog
        outfile = tkFileDialog.asksaveasfile()
        if outfile:
            # save to file
            np.savetxt(outfile, self.map_model.grid, delimiter=',', fmt='%.4f')

    def add_to_polygon(self, event):
        """
        This function adds a point to the polygon (model) and draws a circle

        :param event: Mouse click event.
        """
        # convert from map coordinates back to lon, lat
        lon, lat = self.map_view.map(event.xdata, event.ydata, inverse=True)
        # add coordinates to polygon field of map model.
        self.map_model.polygon = np.vstack([self.map_model.polygon,
                                           [lon, lat]])
        # create new marker at click location.
        p = self.map_view.map.plot(event.xdata, event.ydata,
                                   marker='o', color='r')
        self.map_view.points.append(p[0])
        # draw line if len(polygon) > 1.
        if len(self.map_model.polygon) > 1:
            # draw great circle
            gc = self.map_view.map.drawgreatcircle(
                self.map_model.polygon[-2, 0],
                self.map_model.polygon[-2, 1],
                self.map_model.polygon[-1, 0],
                self.map_model.polygon[-1, 1],
                color='r')
            self.map_view.polylines.append(gc[0])
        # redraw canvas.
        self.map_view.canvas.draw()
        # changed polygon, therefore polygon is not finished
        self.map_model.is_finished = False

    def remove_from_polygon(self, event):
        """
        This function removes the last point from the polygon (model) and
        removes the corresponding line and circle (view).

        :param event: Tkinter event.
        """
        # remove from model
        self.map_model.polygon = self.map_model.polygon[0:-1, :]
        # remove circle and line
        if len(self.map_view.points) > 0:
            self.map_view.points[-1].remove()
            self.map_view.points = self.map_view.points[:-1]
        if self.map_view.polylines > 0:
            self.map_view.polylines[-1].remove()
            self.map_view.polylines = self.map_view.polylines[:-1]
        # redraw
        self.map_view.canvas.draw()

    def select_polygon(self, event):
        """
        This function is invoked if the user clicks on the UI button to select
        a polygon.

        :param event: Tkinter event.
        """
        # connect mouse press in map_view to event handler
        self.psel_cid = self.map_view.canvas.callbacks.connect(
                                'button_press_event', self.mouse_click_handler)

    def finish_polygon(self, event):
        """
        This function is invoked if the user clicks on the UI button to reset
        the polygon.

        :param event: Tkinter event.
        """
        # add last closing segment to plot and list
        gc = self.map_view.map.drawgreatcircle(self.map_model.polygon[-1, 0],
                                               self.map_model.polygon[-1, 1],
                                               self.map_model.polygon[0, 0],
                                               self.map_model.polygon[0, 1],
                                               color='r')
        self.map_view.polylines.append(gc[0])
        self.map_view.canvas.draw()
        # get fine discretization of polygon from matplotlib great circles
        fine_poly = np.array([]).reshape(0, 2)
        all_points = np.array([]).reshape(0, 2)
        for line in self.map_view.polylines:
            # get points of this segment
            new_segment = np.array([line._x, line._y]).T
            all_points = np.vstack([all_points, new_segment])
            fine_poly = np.vstack([fine_poly, new_segment])
        all_points = np.array(self.map_view.map(all_points[:, 0],
                                               all_points[:, 1],
                                               inverse=True)).T
        # save selected points as csv
        np.savetxt('selected_points.csv', all_points, delimiter=',')
        # draw polygon patch
        self.map_view.poly_patch = Polygon(fine_poly, color='r', alpha=0.7)
        self.map_view.axes.add_patch(self.map_view.poly_patch)
        # store final fine polygon into polygon field of model
        self.map_model.polygon = np.array(self.map_view.map(fine_poly[:, 0],
                                                            fine_poly[:, 1],
                                                            inverse=True)).T
        # disconnect mouse button event
        self.map_view.canvas.callbacks.disconnect(self.psel_cid)
        # redraw canvas
        self.map_view.canvas.draw()
        # flag to indicate polygon is finished
        self.map_model.is_finished = True
        # ask to filter model
        if tkMessageBox.askyesno("Filter",
                                 "Filter source model using this polygon?"):
            self.filter_source_model()

    def discretize_polygon(self, event):
        """
        This functions discretizes the selected polygon.

        :param event:
        :return:
        """
        # ask for discretization
        discr_step = tkSimpleDialog.askstring(
                                            "Polygon Discretization",
                                            "Enter grid discretization (deg):")
        discr_step = float(discr_step)

        # get minimum and maximum coordinates of polygon
        minlon = self.map_model.polygon[:, 0].min()
        maxlon = self.map_model.polygon[:, 0].max()
        minlat = self.map_model.polygon[:, 1].min()
        maxlat = self.map_model.polygon[:, 1].max()

        # create grid coordinates
        x, y = np.meshgrid(np.arange(start=minlon,
                                     stop=maxlon,
                                     step=discr_step),
                           np.arange(start=minlat,
                                     stop=maxlat,
                                     step=discr_step))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        points2 = shpgeo.MultiPoint(points)

        # find grid points inside polygon
        poly = shpgeo.Polygon(self.map_model.polygon)
        is_inside = map(poly.contains, points2)
        self.map_model.grid = points.compress(is_inside, axis=0)
        # convert to map coordinates
        grid = np.array(self.map_view.map(self.map_model.grid[:, 0],
                                          self.map_model.grid[:, 1])).T
        # replace old polygon plot py discretization
        self.reset_polygon(event)
        # plot points
        self.map_view.map.scatter(grid[:, 0], grid[:, 1], color='r',
                                  marker='.', s=0.3)
        # redraw canvas
        self.map_view.canvas.draw()
        # ask to save grid as CSV
        if tkMessageBox.askyesno("Save grid", "Save this grid as .csv file?"):
            self.save_grid(event)
            # TODO: filter model in background

    def reset_polygon(self, event):
        """
        Reset the polygon in the model and remove the patch from the view.

        :param event: Tkinter event.
        """
        # remove from model
        self.map_model.reset_polygon()
        # remove polygon patch from map
        if self.map_view.poly_patch:
            self.map_view.poly_patch.remove()
            self.map_view.poly_patch = None
        # remove circles and lines
        while len(self.map_view.points) > 0:
            self.map_view.points[-1].remove()
            self.map_view.points = self.map_view.points[:-1]
        while len(self.map_view.polylines) > 0:
            self.map_view.polylines[-1].remove()
            self.map_view.polylines = self.map_view.polylines[:-1]
        # redraw
        self.map_view.canvas.draw()
        # set flag to to indicate polygon selection is not finished
        self.map_model.is_finished = False

    def analyze_zone(self, event):
        """
        Enable clicking on a zone polygon to open statistical plots.

        :param event:
        :return:
        """
        # connect mouse press in map_view to event handler
        self.sc_cid = self.map_view.canvas.callbacks.connect(
                        'button_press_event', self.analyze_zone_handler)

    def analyze_zone_handler(self, event):
        """
        Determine plot to open.

        :param event: TKinter mouse click event.
        :return:
        """
        # convert from map coordinates back to lon, lat
        lon, lat = self.map_view.map(event.xdata, event.ydata, inverse=True)
        # use source model converter to plot stream of point sources falling
        # into this zone
        self.source_model_converter.plot_min_max_mag_hist(lon, lat)
        self.map_view.canvas.callbacks.disconnect(self.sc_cid)

    def analyze_source_handler(self, event):
        """
        Find closest source and open plots.

        :param event: TKinter mouse click event.
        :return:
        """
        # convert from map coordinates back to lon, lat
        if event.mouseevent.dblclick:
            # plot all sources that were clicked on within tolerance
            for i in event.ind:
                self.nrml_model_container.plot_source_mfd(i)

    def mouse_click_handler(self, event):
        """
        Handled mouse click event into map, adds or removes point of polygon

        :param event: Mouse click event.
        """
        if event.button == 1:
            # left mouse button clicked, add point to polygon.
            self.add_to_polygon(event)
        elif event.button == 3:
            # right mouse button clicked
            # remove entire polygon if finished, otherwise remove last point
            # from polygon.
            if self.map_model.is_finished:
                self.reset_polygon(event)
            else:
                self.remove_from_polygon(event)

    def filter_source_model(self):
        """
        Assigns all ``sources`` inside polygon to valid_sources.
        """
        # again graphical file selection to be able to select non-reduced model
        self.nrml_infile = tkFileDialog.askopenfile()
        if self.nrml_infile:
            # parse sources and put into model field
            self.map_model.sources = read_nodes(
                                            self.nrml_infile,
                                            lambda elem: "Source" in elem.tag,
                                            nrml.nodefactory["sourceModel"])
            # update statusbar
            self.map_view.statusbar.set("Loaded hazard model " +
                                        self.nrml_infile.name + ".")
        self.map_model.valid_sources = []
        # loop over all sources
        i_source = 1
        for source in self.map_model.sources:
            # check if sources in this node are in polygon and distance
            is_in_distance = filter_source.check_source_in_distance(
                                                    source,
                                                    self.map_model.polygon,
                                                    self.map_model.distance,
                                                    self.map_view.statusbar)
            if is_in_distance:
                self.map_model.valid_sources.append(source)
            # update status bar
            i_source += 1
            self.map_view.statusbar.set(
                "Filtering in progress: %i / %i sources inside selected area."
                % (len(self.map_model.valid_sources), i_source))
        # update status bar
        self.map_view.statusbar.set(
                "Filtering finished: %i / %i sources inside selected area."
                % (len(self.map_model.valid_sources), i_source))
        # TODO: update plot of sources

if __name__ == '__main__':
    # initialize and run ui controller
    ui = UiController()
    ui.run()
