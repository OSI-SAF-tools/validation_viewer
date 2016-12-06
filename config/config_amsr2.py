path_to_data = '/home/jol/Documents/Data/amsr2/validation/' # Path to the root folder containing the data
path_to_plots = '/data/jol/osisaf/amsr2/plots/'
path_area_config = '/home/jol/Documents/Programs/sources/validation_viewer/config/areas.cfg'
scaling = 1 # Values are scaled from 0 to 1 -> 0 to 100
bin_intervals = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
             (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
vmin, vmax = 0, 100
#RESULTS_PATHS_NH = {'amsr2_validation_NH_2016-04-26_results.csv': 'OSI Algorithm',
#                   'amsr2_validation_NH_tud_2016-04-26_results.csv': 'TUD Algorithm'}
#RESULTS_PATHS_SH = {'amsr2_validation_SH_2016-04-26_results.csv': 'OSI Algorithm',
#                    'amsr2_validation_SH_tud_2016-04-26_results.csv': 'TUD Algorithm'}
RESULTS_PATHS = {'amsr2_validation_NH_2016-05-17_results.csv':    ('OSI Algorithm', 'Northern Hemisphere'),
                 'amsr2_validation_NH_tud_2016-05-17_results.csv':('TUD Algorithm', 'Northern Hemisphere'),
                 'amsr2_validation_SH_2016-05-17_results.csv':    ('OSI Algorithm', 'Southern Hemisphere'),
                 'amsr2_validation_SH_tud_2016-05-17_results.csv':('TUD Algorithm', 'Southern Hemisphere')}