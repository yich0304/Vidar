import sys
sys.path.insert(0, './tools/analysis_tools')

import mmcv
from nuscenes.nuscenes import NuScenes
import visual

# Load nuScenes and set it globally in visual module
nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=True)
visual.nusc = nusc

# Load your results
results = mmcv.load('test/bevformer_base_vidar/Sat_Dec__6_02_20_25_2025/pts_bbox/results_nusc.json')

# Visualize first 10 samples
sample_tokens = list(results['results'].keys())
for i in range(min(10, len(sample_tokens))):
    visual.render_sample_data(sample_tokens[i], pred_data=results, out_path=f'vis_base_random_{i}')
    print(f'Saved visualization {i}')
