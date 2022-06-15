# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json_tricks as json
from collections import OrderedDict

import numpy as np
from scipy.io import loadmat, savemat

logger = logging.getLogger(__name__)


from dataset.JointsDataset import JointsDataset

class PandoraDataset(JointsDataset):
    
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        
        self.num_joints = 17
        self.train_image_folders = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                                    '11', '12', '13', '14', '15','16', '17')
        self.val_image_folders = ('18', '19', '20', '21', '22')
        
        self.records = ('base_1_ID', 'base_2_ID', 'free_1_ID', 'free_2_ID', 'free_3_ID')
        
        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))
        
        
    def _get_db(self):
        
        # Inintialize db
        gt_db = []
        
        # Create train/val split
        for folder in self.train_image_folders:
            for video_type in self.records:
                file_name = os.path.join(self.root, folder, video_type, 'data.json')
                
                with open(file_name) as anno_file:
                    anno = json.load(anno_file)
                    
                for a in anno:
                    # Get image name
                    image_name = "{:06d}_RGB.png".format(a["frame_num"])
                    
                    # Get center
                    c = 0
                    # Get Scale
                    s = 0
                    
                    if self.image_set != 'test':
                    
                        # Get 2D joints coordinates
                        joints_2d = np.array(a['jointsRGB'])
                        # Get 3D joints coordinates
                        joints_3d = np.array(a['joints3D'])
                        # Get joint visibility
                        joints_vis = np.array(a['joints_vis'])
                        joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
                        joints_3d_vis[:, 0] = joints_vis
                        joints_3d_vis[:, 1] = joints_vis
                        joints_3d_vis[:, 2] = joints_vis
                        
            # Fill db with the given annotations
                    gt_db.append(
                        {
                            'image': os.path.join(os.path.dirname(file_name), 'RGB', image_name),
                            'center': c,
                            'scale': s,
                            'joints_2d': joints_2d,
                            'joints_3d': joints_3d,
                            'joints_3d_vis': joints_3d_vis,
                            'filename': '',
                            'imgnum': 0,
                        }
                    )
        
        return gt_db
    
    
    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        rank = cfg.RANK

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                logger.error('Fail to make {}'.format(res_folder))

        res_file = os.path.join(
            res_folder, 'keypoints_{}_results_{}.json'.format(
                self.image_set, rank)
        )
        
        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0
        
        # Compute PCK@0.5 for 2D coordinates
        SC_BIAS = 0.6
        threshold = 0.5
        
        
        dataset_joints = {
            'nose'
        }
        
        # Calculate 3D PCK for 3D coordinates:
        
