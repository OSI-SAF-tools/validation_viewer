"""
Usage:
    plots_daily.py <plot_function> <output_directory>

Options:
  -h --help     Show this screen.
  --version     Show version..
"""

from __future__ import division

import numpy as np
from numpy import ma
import os
import sys
import traceback
from pandas import *
import pyresample as pr
import h5py
import functools
from multiprocessing import Pool
from docopt import docopt

import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '..'))


class ValidationPlots(object):

    bin_intervals = 100*np.array([(0.00, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
                                  (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.00)])
    bin_edges = np.sort(unique(np.array(bin_intervals).flatten()))
    _hemisphere, hemisphere_old = 'NH', 'NH'

    def __init__(self, path_to_file, path_area_config, projection):
        self.path_to_file = path_to_file
        self.path_area_config = path_area_config
        self.projection = projection
        self.satellite, self.reference = None, None  # Set in read_data()
        self.dates = self.read_data()
        self.df = None # Set in dataframe()
        self.dateidx = None
        self.dateidx_old = None

        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = None, ((None, None),(None, None))
        self.palette1 = None
        self.palette2 = None
        self.bmapax1 = None
        self.bmapax2 = None
        self.bmapax3 = None
        self.bmapax4 = None
        self.setup_plot_both_maps_with_anomaly()
        self.length = self.satellite.shape[-1]

    def __iter__(self):
        for dateidx, date in enumerate(self.dates):
            self.dateidx = dateidx
            self.date = date
            yield self
            try:
                plt.close('all')
                # plt.close(fig)
            except NameError:
                print('Could not close figure')
                pass

    @property
    def hemisphere(self):
        return self._hemisphere

    @hemisphere.setter
    def hemisphere(self, hem):
        self._hemisphere = hem
        if self._hemisphere != self.hemisphere_old:
            self.hemisphere_old = self._hemisphere
            print('Reading {0} data'.format(self._hemisphere))
            self.dates = self.read_data()
            self.setup_plot_both_maps_with_anomaly()
            self.length = self.satellite.shape[-1]

    def read_hdf5(self, ref_sat, algo='data'):
        hdf5 = h5py.File(self.path_to_file, 'r')
        data = hdf5['maps']['data'][self._hemisphere][ref_sat]
        return data.attrs['dates'], data

    def get_ref_sat(self, dateidx):
        if dateidx == 'all':
            # Get all the data
            dateidx = slice(None)
        sat = self.satellite[:, :, dateidx]
        ref = self.reference[:, :, dateidx]
        return ref, sat

    def read_data(self):
        sat_dates, satellite = self.read_hdf5('satellite')
        ref_dates, reference = self.read_hdf5('reference')
        assert all(sat_dates == ref_dates)

        satellite = ma.masked_invalid(satellite[:])
        reference = ma.masked_invalid(reference[:])
        mask = satellite.mask | reference.mask
        satellite.mask = mask
        reference.mask = mask
        self.satellite = ma.masked_invalid(satellite)
        self.reference = ma.masked_invalid(reference)
        return sat_dates

    def dataframe(self, dateidx=None):
        if dateidx is None:
            dateidx = self.dateidx
        if isinstance(self.df, DataFrame) or (self.dateidx_old != dateidx):
            self.dateidx_old = dateidx
            ref, sat = self.get_ref_sat(dateidx)
            ref = ref[np.logical_not(ref.mask)]
            sat = sat[np.logical_not(sat.mask)]
            dd = {'reference': ref.data.flatten(),
                  'satellite': sat.data.flatten()}
            df = DataFrame(data=dd)
            bins_sat = cut(df['satellite'], self.bin_edges, right=True, include_lowest=True)
            bins_reference = cut(df['reference'], self.bin_edges, right=True, include_lowest=True)
            df['bins_sat'] = list(bins_sat)
            df['bins_reference'] = list(bins_reference)
            self.df = df.dropna()
            return self.df
        else:
            return self.df

    def pivot_table(self, dateidx, **kwargs):
        df = self.dataframe(dateidx)
        return pivot_table(df, values='satellite', index=['bins_sat'], columns=['bins_reference'],
                           aggfunc=len, **kwargs)

    # def plot_stats(self, parameter, width, height, algo, loc=2):
    #     df = self.dataframe()
    #     matplotlib.rcParams.update({'font.size': 10})
    #     title = parameter.replace('_', ' ').title() + ', ' + algo
    #     df['nh'][parameter].plot(style='.-', title=title, figsize=(width, height))
    #     df['sh'][parameter].plot(style='.-', title=title, figsize=(width, height))
    #     grid()

    def plot_hemisphere(self, hm, ax):
        area_def = pr.utils.load_area(self.path_area_config, '{0}_{1}'.format(self.projection, hm))
        bmap = pr.plot.area_def2basemap(area_def)
        bmap.ax = ax
        bmap.drawcoastlines()
        # draw parallels.
        parallels = np.arange(-360, 360, 10)
        bmap.drawparallels(parallels, labels=[0, 0, 0, 0], fontsize=10)
        # draw meridians
        meridians = np.arange(0., 360., 10)
        bmap.drawmeridians(meridians, labels=[0, 0, 0, 0], fontsize=10)
        return bmap

    def setup_plot_both_maps_with_anomaly(self):

        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(11, 11))
        self.palette1 = plt.cm.gist_ncar
        self.palette1.set_bad(color='0.8')  # Masked values are grey
        self.palette2 = plt.cm.seismic
        self.palette2.set_bad(color='0.8')
        self.bmapax1 = self.plot_hemisphere(self._hemisphere, self.ax1)
        self.bmapax2 = self.plot_hemisphere(self._hemisphere, self.ax2)
        self.bmapax3 = self.plot_hemisphere(self._hemisphere, self.ax3)
        self.bmapax4 = self.plot_hemisphere(self._hemisphere, self.ax4)

    def plot_both_maps_with_anomaly(self, dateidx=None, vmin=0, vmax=100, origin='upper'):
        """
        For plotting in when running in parallel
        """
        if dateidx is None:
            dateidx = self.dateidx

        self.Date = self.dates[dateidx]
        ref, sat = self.get_ref_sat(dateidx)
        self.fig.suptitle('Sea Ice Concentration (OSI SAF vs NIC) Comparison\n{0}'.format(self.Date),
                     fontsize=14, verticalalignment='top')

        im1 = self.bmapax1.imshow(sat, origin=origin, cmap=self.palette1, vmin=0, vmax=vmax)
        self.bmapax1.ax.set_title('OSI SAF Percentage SIC', fontsize=10)

        im2 = self.bmapax2.imshow(ref, origin=origin, cmap=self.palette1, vmin=0, vmax=vmax)
        self.bmapax2.ax.set_title('NIC Percentage SIC', fontsize=10)

        ic_diff = (ref - sat)

        data = ma.array(ic_diff, mask=(ref > 10))
        im3 = self.bmapax3.imshow(data, origin=origin, cmap=self.palette2, vmin=-25, vmax=25)
        self.bmapax3.ax.set_title("SIC Anomaly ('NIC' - 'OSI SAF') where the NIC shows Water", fontsize=10)

        data = ma.array(ic_diff, mask=(ref <= 90))

        im4 = self.bmapax4.imshow(data, origin=origin, cmap=self.palette2, vmin=-25, vmax=25)
        self.bmapax4.ax.set_title("SIC Anomaly ('NIC' - 'OSI SAF') where the NIC shows Ice", fontsize=10)

        # plt.tight_layout()
        cbar_ax1 = self.fig.add_axes([0.04, 0.54, 0.025, 0.35])
        self.fig.colorbar(im1, cax=cbar_ax1)
        cbar_ax2 = self.fig.add_axes([0.04, 0.110, 0.025, 0.35])
        self.fig.colorbar(im3, cax=cbar_ax2)
        
        return self.fig

    def plot_both_maps_with_anomaly_i(self, dateidx=None, vmin=0, vmax=100):
        """
        For Jupyter Interactive Plotting
        """
        plt.close('all')
        self.setup_plot_both_maps_with_anomaly()
        self.plot_both_maps_with_anomaly(dateidx, vmin, vmax, origin='upper')

    def plot_kde(self, dateidx=None):
        if dateidx is None:
            dateidx = self.dateidx
        import seaborn as sb
        ref, sat = self.get_ref_sat(dateidx)
        notmasked = np.logical_not(ref.mask.flatten())
        ref, sat = ref.flatten()[notmasked], sat.flatten()[notmasked]
        take = 10000 if len(ref) >= 10000 else len(ref)
        ch = np.random.choice(len(ref), take, replace=False)
        ref, sat = ref[ch], sat[ch]
        sb.kdeplot(ref, sat, shade=True, kernel='cos')
        plt.scatter(ref, sat, s=0.5)
        return plt

    def scatter(self, dateidx=None):
        if dateidx is None:
            dateidx = self.dateidx
        df = self.dataframe(dateidx)
        df.plot(kind='scatter', x='reference', y='satellite', s=0.001, figsize=(4.5, 4.5))
        plt.plot([0, 100],[0, 100])
        plt.grid()
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.xlabel('val. ref. emissivity')
        plt.ylabel('product emissivity')
        t = 'Scatter Plot of Ice Concentration in the\n {0} Hemisphere on {1}'
        plt.title(t.format(self._hemisphere, self.dates[dateidx]))

    def hex_bin(self, dateidx=None):
        if dateidx is None:
            dateidx = self.dateidx
        df = self.dataframe(dateidx)
        df.plot(kind='hexbin', x='reference', y='satellite', gridsize=30)
        # plot([0, 100], [0,100], '--')

    def histogram(self, dateidx=None):
        if dateidx is None:
            dateidx = self.dateidx
        df = self.dataframe(dateidx)
        d = np.nan_to_num(df['satellite'])
        bins = [0, 81, 95, 100]
        bins = np.linspace(0, 101, 6)
        np.hist(d[d>0], bins=bins, alpha=0.5)
        plt.title('Satellite')
        plt.grid()
        # show()
        d = np.nan_to_num(df['reference'])
        np.hist(d[d>0], bins=bins, alpha=0.5)
        plt.title('Reference')
        plt.grid()

    def heat_map(self, dateidx=None, colorscale=None):
        global fig
        if dateidx is None:
            dateidx = self.dateidx
        import seaborn as sb
        from matplotlib.colors import LogNorm

        sb.set(font_scale=1.2)
        pt = self.pivot_table(dateidx)
        pt_norm = 100 * pt / pt.sum().sum()
        idx = Index(reversed([u'[0, 10]', u'(10, 20]', u'(20, 30]', u'(30, 40]', u'(40, 50]',
                     u'(50, 60]', u'(60, 70]', u'(70, 80]', u'(80, 90]', u'(90, 100]']),
                    dtype='object', name=u'OSI SAF SIC')
        cols = Index([u'[0, 10]', u'(10, 20]', u'(20, 30]', u'(30, 40]', u'(40, 50]',
                      u'(50, 60]', u'(60, 70]', u'(70, 80]', u'(80, 90]', u'(90, 100]'],
                     dtype='object', name=u'Ice Chart SIC')
        pt_norm = pt_norm.reindex(index=idx, columns=cols).fillna(0)
        fig = plt.figure(figsize=(10, 9))
        vmin, vmax = 0.1, 60

        if colorscale == 'LogNorm':
            norm = LogNorm(vmin=vmin, vmax=vmax)
            vmin, vmax = 0.1, 60
            cbar = True
        elif colorscale == None:
            norm = None
            vmin, vmax = 0, 60
            cbar = True
        if colorscale == 'Norm':
            norm = None
            cbar = False
            normheatmap = lambda ax: sb.heatmap(100*pt_norm/pt_norm.max(axis=0),
                                                annot=False, fmt='.2f', linewidths=.5, cmap=plt.cm.plasma,
                                                square=True, norm=norm, annot_kws={"size": 10}, cbar=True, ax=ax)

        ax = sb.heatmap(pt_norm, annot=True, fmt='.2f', linewidths=.5, cmap=plt.cm.plasma, square=True,
                   vmin=vmin, vmax=vmax, norm=norm, annot_kws={"size": 10}, cbar=cbar)
        if colorscale == 'Norm':
            normheatmap(ax)
        plt.title('{0}'.format(self.dates[dateidx]), fontweight='bold', fontsize=18)
        return fig

    def heat_map_log(self, dateidx=None):
        return self.heat_map(dateidx, colorscale='LogNorm')

    def heat_map_norm(self, dateidx=None):
        return self.heat_map(dateidx, colorscale='Norm')


def make_plot(plot_type, out_dir, hm, i):
    "Generates images for all days"

    try:
        fname = '{0}_{1}_{2}.png'.format(plot_type, hm, i)
        out_path = os.path.join(out_dir, fname)
        if not os.path.isfile(out_path):
            print('Making: ' + fname)
            figure = getattr(vplots, plot_type)(i)
            figure.savefig(out_path)
            plt.close(figure)
        else:
            print('Skipping: ' + fname)
    except ValueError:
        pass
    except Exception as e:
        print(traceback.format_exc())


def make_video(plot_function, output_directory):

    global vplots

    import config.config_osi450 as cfg

    vplots = ValidationPlots(cfg.path_to_hdf5, cfg.path_area_config, 'EASE2')

    for hm in ['NH', 'SH']:
        vplots.hemisphere = hm
        pg = functools.partial(make_plot,
                               plot_function,
                               output_directory,
                               vplots.hemisphere)
        pool = Pool(processes=4)
        pool.map(pg, range(0, vplots.length))

        cmd = """
        cd {0}
        rm {2}/*
        x=1
        for i in $(ls {1}_{2}*png | sort -t _ -k 5 -g)
            do
                echo $i
                counter=$(printf %04d $x)
                ln -s {0}/$i {0}/{2}/img"$counter".png
                x=$(($x+1))
            done
        """
        print(cmd.format(output_directory, plot_function, hm))
        os.system(cmd.format(output_directory, plot_function, hm))

        cmd = "avconv -i {0}/{2}/img%04d.png -r 25 -c:v libx264 -crf 25 -pix_fmt yuv420p {0}/{1}_{2}.mp4"
        os.system(cmd.format(output_directory, plot_function, hm))


if __name__ == '__main__':
    plt.switch_backend('cairo')
    args = docopt(__doc__)
    make_video(args['<plot_function>'], args['<output_directory>'])
