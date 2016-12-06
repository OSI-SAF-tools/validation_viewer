from __future__ import division

from pylab import *
import os
import traceback
from functools import partial
import glob
from pandas import *
import matplotlib
import matplotlib.pyplot as plt
from pandas import datetime
import pyresample as pr
import h5py


class ValidationInfo(object):

    bin_intervals = 100*np.array([(0.00, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
                                  (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.00)])
    bin_edges = sort(unique(array(bin_intervals).flatten()))
    _hemisphere, hemisphere_old = 'NH', 'NH'

    def __init__(self, path_to_file, path_area_config, projection):
        self.path_to_file = path_to_file
        self.path_area_config = path_area_config
        self.projection = projection
        self.satellite, self.reference = None, None # Set in read_data()
        self.dates = self.read_data()
        self.df = None # Set in dataframe()
        self.dateidx = None
        self.dateidx_old = None

    def __iter__(self):
        for dateidx, date in enumerate(self.dates):
            self.dateidx = dateidx
            self.date = date
            yield self
            try:
                plt.close(fig)
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

    def read_hdf5(self, hemisphere, ref_sat, algo='data'):
        hdf5 = h5py.File(self.path_to_file, 'r')
        data = hdf5['maps']['data'][hemisphere][ref_sat]
        return data.attrs['dates'], data

    def get_ref_sat(self, dateidx):
        if dateidx == 'all':
            # Get all the data
            dateidx = slice(None)
        sat = self.satellite[:, :, dateidx]
        ref = self.reference[:, :, dateidx]
        return ref, sat

    def read_data(self):
        sat_dates, satellite = self.read_hdf5(self._hemisphere, 'satellite')
        ref_dates, reference = self.read_hdf5(self._hemisphere, 'reference')
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



class ValidationPlots(ValidationInfo):

    def plot_hemisphere(self, hm, ax, data, **imshowargs):
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
        im = bmap.imshow(data, origin='upper', **imshowargs)
        return im, bmap.ax

    def plot_map_with_anomaly(self, dateidx=None, vmin=0, vmax=100):

        # The plots are not automatically cleared if the function is repeatedly called.
        # This stops RAM being used up.
        global bmapax1, bmapax2, fig
        try:
            bmapax1.cla()
            bmapax2.cla()
        except NameError:
            pass

        if dateidx is None:
            # self.dateidx is set by the __iter__ method if not given in the function
            dateidx = self.dateidx
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(11, 11))
        palette1 = plt.cm.gist_ncar
        palette1.set_bad(color='k', alpha=0.2)  # Masked values are grey
        palette2 = plt.cm.seismic
        palette2.set_bad(color='k', alpha=0.2)
        ref, sat = self.get_ref_sat(dateidx)
        im1, bmapax1 = self.plot_hemisphere(self._hemisphere, ax1, sat, cmap=palette1, vmin=0, vmax=vmax)
        Date = self.dates[dateidx]
        bmapax1.set_title('Date: {0}\n\nOSI SAF SIC'.format(Date), fontsize=14)
        im2, bmapax2 = self.plot_hemisphere(self._hemisphere, ax2, (ref - sat),
                                            cmap=palette2, vmin=-1 * vmax, vmax=vmax)
        bmapax2.set_title("Spatial Anomaly in SIC ('Ice-Chart' - 'OSI SAF')", fontsize=14)
        plt.tight_layout()
        cbar_ax1 = fig.add_axes([0.71, 0.52, 0.025, 0.33])
        fig.colorbar(im1, cax=cbar_ax1)
        cbar_ax2 = fig.add_axes([0.71, 0.034, 0.025, 0.33])
        fig.colorbar(im2, cax=cbar_ax2)
        return fig

    def plot_both_maps_with_anomaly(self, dateidx=None, vmin=0, vmax=100):

        # The plots are not automatically cleared if the function is repeatedly called.
        # This stops RAM being used up.
        global bmapax1, bmapax2, bmapax3, bmapax4, fig
        try:
            bmapax1.cla()
            bmapax2.cla()
            bmapax3.cla()
            bmapax4.cla()
        except NameError:
            print('Could not clear axis')
            pass

        if dateidx is None:
            # self.dateidx is set by the __iter__ method if not given in the function
            dateidx = self.dateidx
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(11, 11))
        palette1 = plt.cm.gist_ncar
        palette1.set_bad(color='k', alpha=0.2)  # Masked values are grey
        palette2 = plt.cm.seismic
        palette2.set_bad(color='k', alpha=0.2)
        ref, sat = self.get_ref_sat(dateidx)
        Date = self.dates[dateidx]

        fig.suptitle('Sea Ice Concentration (OSI SAF vs NIC) Comparison\n{0}'.format(Date),
                     fontsize=14, verticalalignment='top')

        im1, bmapax1 = self.plot_hemisphere(self._hemisphere, ax1, sat, cmap=palette1, vmin=0, vmax=vmax)
        bmapax1.set_title('OSI SAF Percentage SIC', fontsize=10)

        im2, bmapax2 = self.plot_hemisphere(self._hemisphere, ax2, ref, cmap=palette1, vmin=0, vmax=vmax)
        bmapax2.set_title('NIC Percentage SIC', fontsize=10)

        ic_diff = (ref - sat)

        data = ma.array(ic_diff, mask=(ref > 10))
        im3, bmapax3 = self.plot_hemisphere(self._hemisphere, ax3, data, cmap=palette2, vmin=-25, vmax=25)
        bmapax3.set_title("SIC Anomaly ('NIC' - 'OSI SAF') where the NIC shows Water", fontsize=10)

        data = ma.array(ic_diff, mask=(ref <= 90))
        im4, bmapax4 = self.plot_hemisphere(self._hemisphere, ax4, data, cmap=palette2, vmin=-25, vmax=25)
        bmapax4.set_title("SIC Anomaly ('NIC' - 'OSI SAF') where the NIC shows Ice", fontsize=10)

        # plt.tight_layout()
        cbar_ax1 = fig.add_axes([0.04, 0.548, 0.025, 0.35])
        fig.colorbar(im1, cax=cbar_ax1)
        cbar_ax2 = fig.add_axes([0.04, 0.128, 0.025, 0.35])
        fig.colorbar(im3, cax=cbar_ax2)
        
        return fig

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
        plot([0, 100],[0, 100])
        grid()
        xlim(0, 100)
        ylim(0, 100)
        xlabel('val. ref. emissivity')
        ylabel('product emissivity')
        t = 'Scatter Plot of Ice Concentration in the\n {0} Hemisphere on {1}'
        title(t.format(self._hemisphere, self.dates[dateidx]))

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
        d = nan_to_num(df['satellite'])
        bins = [0, 81, 95, 100]
        bins = linspace(0, 101, 6)
        hist(d[d>0], bins=bins, alpha=0.5)
        title('Satellite')
        grid()
        #show()
        d = nan_to_num(df['reference'])
        hist(d[d>0], bins=bins, alpha=0.5)
        title('Reference')
        grid()

    def heat_map(self, dateidx=None, colorscale=None):
        global fig
        if dateidx is None:
            dateidx = self.dateidx
        import seaborn as sb
        from matplotlib.colors import LogNorm

        sb.set(font_scale=1.4)
        pt = self.pivot_table(dateidx)
        pt_norm = 100 * pt / pt.sum().sum()
        idx = Index(reversed([u'[0, 10]', u'(10, 20]', u'(20, 30]', u'(30, 40]', u'(40, 50]',
                     u'(50, 60]', u'(60, 70]', u'(70, 80]', u'(80, 90]', u'(90, 100]']),
                    dtype='object', name=u'OSI SAF SIC')
        cols = Index([u'[0, 10]', u'(10, 20]', u'(20, 30]', u'(30, 40]', u'(40, 50]',
                      u'(50, 60]', u'(60, 70]', u'(70, 80]', u'(80, 90]', u'(90, 100]'],
                     dtype='object', name=u'Ice Chart SIC')
        pt_norm = pt_norm.reindex(index=idx, columns=cols).fillna(0)
        fig = figure(figsize=(10, 9))
        vmin, vmax = 0.1, 60

        if colorscale == 'LogNorm':
            norm = LogNorm(vmin=vmin, vmax=vmax)
            vmin, vmax = 0.1, 60
        elif colorscale == None:
            norm = None
            vmin, vmax = 0, 60
        sb.heatmap(pt_norm, annot=True, fmt='.2f', linewidths=.5, cmap=plt.cm.plasma, square=True,
                   vmin=vmin, vmax=vmax, norm=norm, annot_kws={"size": 10})
        title('{0}'.format(self.dates[dateidx]), fontweight='bold', fontsize=18)
        return fig

    def heat_map_log(self, dateidx=None):
        return self.heat_map(dateidx, colorscale='LogNorm')


def plot_genertor(plot_type, path_to_hdf5, out_dir, path_area_config, projection):
    "Generates images for all days"

    vplots = ValidationPlots(path_to_hdf5, path_area_config, projection)
    for hm in ('NH', 'SH'):
        vplots.hemisphere = hm
        for i, vp in enumerate(vplots):
            try:
                fname = '{0}_{1}_{2}.png'.format(plot_type, hm, i)
                out_path = os.path.join(out_dir, fname)
                print(fname)
                if not os.path.isfile(out_path):
                    figure = getattr(vp, plot_type)()
                    figure.savefig(out_path)
                    plt.close(figure)
                del vp
            except ValueError:
                pass
            except Exception as e:
                print(traceback.format_exc())

# imporot_map_with_anolomoly', cfg.path_to_hdf5, out_dir, cfg.path_area_config, 'EASE2')

"""

# %%bash
# cd /data2/validation/amsr2
# wget ftp://ftp.dmi.dk/sat/amsr2_validation/amsr2_val_2015.tar.gz
# mkdir /data2/validation/amsr2/data
# tar -xvf amsr2_val_2015.tar.gz -C /home/jol/Documents/Data/amsr2/validation/

# %%bash
# wget ftp://ftp.dmi.dk/sat/ssmis_emiss/emiss_val_2015.tar.gz
# mkdir data
# tar -xvf emiss_val_2015.tar.gz -C data

## Wait for the above code to download the validation data and extract to the 'data' folder in the working directory.

%%bash
cd /home/jol/Documents/Programs/data_analaysis/validation/plots
convert -loop 0 NH*2015-01*.png NH_emissivity.gif
convert -loop 0 SH*2015-01*.png SH_emissivity.gif



## Create Some Statistics

hm_dic = {'Northern':'NH' , 'Southern':'SH'}
stats_dict = {}
for hemisphere in ['Northern', 'Southern']:
    stats = []
    for date in dates[0:]:
        try:
            df = interact_out(hemisphere, date, 'Dataframe')
            df = df.dropna()
            stats.append((datetime.strptime(date, '%Y-%m-%d'),
                          df['satellite'].mean(),
                          df['reference'].mean(),
                          ))
        except AttributeError:
            pass
    stats_dict[hm_dic[hemisphere]] = array(stats)

matplotlib.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(18, 12))
def plot_err(hem):
    x = stats_dict[hem][:,0]
    y = stats_dict[hem][:,1]
    yerr = stats_dict[hem][:,2]
    ax.errorbar(x, y, yerr=yerr, fmt='o',alpha=0.5)
    plttitle = {'NH': 'Northern' , 'SH':'Southern'}
    title('Mean Difference Between The Product & Reference Emissivity\nwith Std.'
          ' in the Northern (Blue) & Southern (Green) Hemispheres')
    #title('Mean Difference Between The Product & Reference Emissivity with Std.\n'
    #      'In The '+plttitle[hem]+ ' Hemisphere')
    fig.autofmt_xdate()
    ylabel('Emissivity')
plot_err('NH')
#show()
plot_err('SH')
savefig('{0}{1}_{2}{3}'.format(path_to_plots, 'mean', date, '.png'))
#show()

matplotlib.rcParams.update({'font.size': 10})
def plot2(hem):
    fig, ax = plt.subplots()
    date = stats_dict[hem][:,0]
    sat = stats_dict[hem][:,1]
    ref = stats_dict[hem][:,2]

    ax.plot(date, sat,'.-')
    ax.plot(date, ref,'.-')
    plttitle = {'NH': 'Northern' , 'SH':'Southern'}
    title('The Mean Emissivity of the Product & Reference'
          '\nin the %s Hemisphere' % plttitle[hem])
    #title('Mean Difference Between The Product & Reference Emissivity with Std.\n'
    #      'In The '+plttitle[hem]+ ' Hemisphere')
    fig.autofmt_xdate()
    ylabel('Emissivity')
plot2('NH')
legend(['Product','Reference'],loc=3)
grid()
savefig('{0}{1}_{2}{3}'.format(path_to_plots, 'mean_NH', date, '.png'),dpi=120)
plot2('SH')
legend(['Product','Reference'],loc=8)
grid()
savefig('{0}{1}_{2}{3}'.format(path_to_plots, 'mean_SH', date, '.png'),dpi=120)


for i, date in enumerate(dates):
    try:
        fig = interact_out(i, 'figure')
        p_paths = os.path.join(path_to_plots,'{0}_{1}{2}{3}'.format(Hemisphere, date, SUFFIX,'.png'))
        print(p_paths)
        fig.savefig(p_paths, dpi=200, bbox_inches='tight')
    except AttributeError:
        print "ERROR"
        pass

"""