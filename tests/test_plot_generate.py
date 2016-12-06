from __future__ import division

import viewer.plots_daily as pd
import config.config_osi450 as cfg

pd.plot_genertor('plot_map_with_anolomoly', cfg.path_to_hdf5,
                 '/data/jol/temp/pngs/plot_map_with_anolomoly/', cfg.path_area_config, 'EASE2')

pd.plot_genertor('heat_map', cfg.path_to_hdf5,
                 '/data/jol/temp/pngs/heat_map/', cfg.path_area_config, 'EASE2')

