# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder
from .single_stage import SingleStage3DDetector
import numpy as np
import open3d as o3d


@DETECTORS.register_module()
class PillarGridShare(SingleStage3DDetector):
    r"""PillarGrid with shared weights for pillar feature encoder"""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 fusion_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(PillarGridShare, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.fusion_encoder = builder.build_fusion_encoder(fusion_encoder)

    def extract_feat(self, points, img_metas=None):
        """Extract features from points."""
        points_02_batch = []
        for i in range(len(img_metas)):
            pts_02_filename = img_metas[i]['pts_filename'].replace('velodyne', 'velodyne_02_transformed')
            # print('img_metas', img_metas)
            # print('pts_02_filename', pts_02_filename)
            points_02 = self.load_02_velodyne(pts_02_filename)
            points_02_batch.append(points_02)

        # print('points_02', points_02[0].shape)
        # print('points', points[0].shape)
        # print('points', points)
        # print('points_02_batch',points_02_batch)
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)


        with torch.no_grad():
            voxels_02, num_points_02, coors_02 = self.voxelize(points_02_batch)
            
            voxel_features_02 = self.voxel_encoder(voxels_02, num_points_02, coors_02)
            batch_size_02 = coors_02[-1, 0].item() + 1
            x_02 = self.middle_encoder(voxel_features_02, coors_02, batch_size_02)
        
        # print('voxelize', voxels.shape)
        # print('voxel_encoder output', voxel_features.shape)
        # print('voxel_encoder_02 output', voxel_features_02.shape)

        """
        voxelize torch.Size([17872, 32, 4])
        voxel_encoder output torch.Size([17872, 64])
        middle_encoder torch.Size([1, 64, 512, 512])
        """



        x_3d = torch.unsqueeze(x, 4)
        x_02_3d = torch.unsqueeze(x_02, 4)

        # print('middle_encoder', x.shape)
        # print('middle_encoder_02', x_02.shape)
        # print('x_3d', x_3d.shape)
        # print('x_02_3d', x_02_3d.shape)

        x_fusion = torch.cat([x_3d, x_02_3d], 4)


        # print('x_fusion', x_fusion[:,63, 250, 200:250, :])
        
        x = self.fusion_encoder(x_fusion)

        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        # print('points02', points02)
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas,  imgs=None, rescale=False):
        """Test function without augmentaiton."""
        # print('simple test, pts_filename', pts_filename)
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]

    def load_02_velodyne(self, pts_02_filename):
        pcd = np.fromfile(pts_02_filename, dtype=np.float32).reshape(-1, 4)
        
        """
        Display onboard point clouds
        """
        # pcd_xyz = pcd[:, :3]
        # # print(pcd_xyz)
        # pcd_o3d = o3d.geometry.PointCloud()
        # pcd_o3d.points = o3d.utility.Vector3dVector(pcd_xyz)
        # o3d.visualization.draw_geometries([pcd_o3d])

        pcd_tensor = torch.from_numpy(pcd).to('cuda:0')
        return pcd_tensor