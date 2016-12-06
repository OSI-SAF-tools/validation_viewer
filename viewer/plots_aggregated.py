from pylab import *
import os
from pandas import *
import matplotlib
from pandas import datetime

from matplotlib import pyplot
# from config.config_amsr2 import *


# data_dir = '/home/jol/Documents/Data/amsr2/validation/' # Path to the root folder containing the data
# output_dir = '/data/jol/osisaf/amsr2/plots/'
# area_config_path = '/home/jol/Documents/Programs/sources/validation_viewer/config/areas.cfg'
# bin_intervals = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
#              (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
# vmin, vmax = 0, 100
# #RESULTS_PATHS_NH = {'amsr2_validation_NH_2016-04-26_results.csv': 'OSI Algorithm',
# #                   'amsr2_validation_NH_tud_2016-04-26_results.csv': 'TUD Algorithm'}
# #RESULTS_PATHS_SH = {'amsr2_validation_SH_2016-04-26_results.csv': 'OSI Algorithm',
# #                    'amsr2_validation_SH_tud_2016-04-26_results.csv': 'TUD Algorithm'}
# results_output = {'amsr2_validation_NH_2016-05-17_results.csv':    ('OSI Algorithm', 'Northern Hemisphere'),
#                  'amsr2_validation_NH_tud_2016-05-17_results.csv':('TUD Algorithm', 'Northern Hemisphere'),
#                  'amsr2_validation_SH_2016-05-17_results.csv':    ('OSI Algorithm', 'Southern Hemisphere'),
#                  'amsr2_validation_SH_tud_2016-05-17_results.csv':('TUD Algorithm', 'Southern Hemisphere')}

class ValidationPlots(object):

    def __init__(self, data_dir, output_dir, area_config_path,
                 bin_intervals, results_output):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.area_config_path = area_config_path
        self.bin_intervals = bin_intervals
        self.results_output = results_output

    def read_summary_stats(self, filepath):
        dateparser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        df = read_csv(filepath, parse_dates=[0,1,2], date_parser=None,
                      index_col=0,na_values='--')
        return df

    def line_plots(self, para, ymin=0, ymax=100):
        # Plot Dimensions
        width = 17
        height = 0.5*(width/1.618)
        df = {}
        #para = 'stddev'
        # para = 'bias'
        vvars = ['water', 'ice', 'intermediate']
        summaries = {}
        for path in self.results_output:
            df = self.read_summary_stats(os.path.join(self.data_dir, path))

            for var in vvars:
                key = var + '_' + para
                print('A min of {0:.2f} occurs on {1} for {2}'.format(df[key].min(),
                                                                      df[key].idxmin(),
                                                                      key
                                                                      ))
                print('A max of {0:.2f} occurs on {1} for {2}'.format(df[key].max(),
                                                                      df[key].idxmax(),
                                                                      key
                                                                      ))
                # if isinstance(RESULTS_PATHS, dict):
                #    title_suffix = '{0} in the {1}'.format(*RESULTS_PATHS[path])
                # else:
                #    title_suffix = ''
                # val.plot_stats(df, var, width, height, title_suffix)
                matplotlib.rcParams.update({'font.size': 10})
                df[key].plot(style='.-', figsize=(width, height), grid=True, linewidth=0.5, markersize=2)
            ll = [self.results_output[path][0], path.split('_')[2]]
            legend(vvars, loc=8, ncol=3, fontsize=8)
            ll.append(para)
            pyplot.title('{0}, {1} Hemisphere'.format(*ll))
            summaries['{0}_{1}'.format(*ll).replace(' ', '')] = df.describe()
            plotpath = os.path.join(self.output_dir, '{0}{1}{2}.png'.format(*ll).replace(' ', ''))
            # ylim(-30, 5)
            ylim(ymin, ymax)
            xlabel('Date')
            ylabel('Percentage ' + para.capitalize())
            print('Saving {0}'.format(plotpath))
            # savefig(plotpath, dpi=300)
            show()

        return df #Panel(summaries)

    def stacked_bar(self, df_10, df_20, ind, ymin=50, ymax=100):
        fig = figure(num=None, figsize=(17, 6), dpi=80, facecolor='w', edgecolor='k')
        ax = pyplot.subplot(111)

        # ind = range(len(df.index))
        width = 6  # the width of the bars: can also be len(x) sequence

        p1 = ax.bar(ind, df_20, width, color='r', edgecolor='none')
        p2 = ax.bar(ind, df_10, width, color='b', edgecolor='none')
        ax.xaxis_date()

        ylim(ymin, ymax)
        grid()
        legend(('Within +/-20%', 'Within +/-10%'), loc=3)
        xlabel('Date')
        ylabel('Percentage Of Grid Points')
        return fig

    def barcharts(self, ymin=0, ymax=100):
        means = []
        for path in self.results_output:
            df = self.read_summary_stats(os.path.join(self.data_dir, path))
            print(df[['within_10pct', 'within_20pct']].describe())
            fig = self.stacked_bar(df['within_10pct'], df['within_20pct'], df.index, ymin, ymax)
            path_split = path.split('_')
            if len(path_split) == 6:
                instrument, what, hem, algo, date, fname = path_split
            elif len(path_split) == 5:
                instrument, what, hem, date, fname = path_split
                algo = 'OSI SAF'
            else:
                print('Expecting the length of the path to be 5 or 6')
            hh = {'NH': 'Northern', 'SH': 'Southern'}[hem]
            means.append((hh, algo, df['within_10pct'].mean(), df['within_20pct'].mean()))
            # print '{0} Hemisphere, {1}'.format(hh, algo)
            pyplot.title('{0} Hemisphere, {1} Algorithm'.format(hh, algo.upper()))
            # savefig('/home/jol/Documents/Programs/data_analaysis/validation/{0}'.format(path.replace('.csv', '.png')), dpi=180)
            pyplot.show()


