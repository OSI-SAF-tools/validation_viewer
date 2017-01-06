from __future__ import division

import viewer.plots_daily as pd
import config.config_osi450 as cfg

# pd.make_video('plot_map_with_anolomoly', cfg.path_to_hdf5,
#                  '/data/jol/temp/pngs/plot_map_with_anolomoly/', cfg.path_area_config, 'EASE2')

pd.make_video('heat_map_norm', '/data/jol/validation/plots/heat_map_norm/')

