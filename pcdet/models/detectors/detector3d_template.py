import os

import torch
import torch.nn as nn

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils.spconv_utils import find_all_spconv_keys
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..model_utils import model_nms_utils


class Detector3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        """
        self:
            class_names={list:1}['stone']
            dataset={StoneDataset:54}
            global_step(待定)
            model_cfg(Model的配置文件信息)
            model_topology={list:8}backbone_3d,map_to_bev_module,pfe,backbone_2d,dense_head,point_head,roi_head
        :return: model的网络架构
        """
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features, # 点特征数量 4 这里是调用函数获取
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,  # 网格大小 （432,496, 1） 直接根据dataset属性获取
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'depth_downsample_factor': self.dataset.depth_downsample_factor  # 下采样因子
        }
        for module_name in self.module_topology:  # vfe,map_to_bev,backbone_2d,dense_head
            # 在Detector3DTemplate生成全部子模块，如果配置文件不存在该子摸块，则直接返回None
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(  #调用self.build_vfe...
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)   # 这里调用nn.Module中的add_module方法，向当前的模块中填加子模块
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict
        # 但是这里只是初始化，并没有调用forward函数
        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
            depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        model_info_dict['backbone_channels'] = backbone_3d_module.backbone_channels \
            if hasattr(backbone_3d_module, 'backbone_channels') else None
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features']
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        """
        CenterHead(
          (shared_conv): Sequential(
            (0): Conv2d(384, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (heads_list): ModuleList(
            (0): SeparateHead(
              (stone): Sequential(
                (0): Sequential(
                  (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU()
                )
                (1): Conv2d(64, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
              (hm): Sequential(
                (0): Sequential(
                  (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU()
                )
                (1): Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
            )
          )
          (hm_loss_func): FocalLossCenterNet()
          (reg_loss_func): RegLossCenterNet()
        )
        """
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict

        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']

        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](
            model_cfg=self.model_cfg.POINT_HEAD,
            input_channels=num_point_features,
            num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg=self.model_cfg.ROI_HEAD,
            input_channels=model_info_dict['num_point_features'],
            backbone_channels=model_info_dict['backbone_channels'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        """  后处理
        1. 根据index和mask进行box和cls进行筛选
        2. 对该帧点云进行NMS
        3. 计算召回率
        4. 返回该batch的预测结果和recall
        Args:
            batch_dict:
                batch_size: 批量大小
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1) # 类别预测
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C) # 边框预测
                cls_preds_normalized: indicate whether batch_cls_preds is normalized # 是否进行归一化
                batch_index: optional (N1+N2+...) # batch索引 eg:[1 1 1 1 1 1 1 2 2 2 2 2....]
                has_class_labels: True/False # 是否有类别标签
                roi_labels: (B, num_rois)  1 .. num_classes # roi标签
                batch_pred_labels: (B, num_boxes, 1) # 批量预测标签
        Returns:
        """
        post_process_cfg = self.model_cfg.POST_PROCESSING  # 后处理配置参数
        batch_size = batch_dict['batch_size']  # batch size
        recall_dict = {}  # 仅仅是一个字典，在逐帧处理中不断更新
        pred_dicts = []   # 预测字典列表
        # 对batch中的每帧点云的预测结果进行后处理
        for index in range(batch_size):  # 1.根据index和mask进行box和cls进行筛选
            if batch_dict.get('batch_index', None) is not None:
                # 如果存在batch_index,则需要根据当前index进行选择，且此时的box的预测维度为2（ N = B * num_boxes , 7+C)
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)  # （N,）
            else:
                # 如果不存在batch_index,且此时的box的预测维度为3:(B, num_boxes, 7+C),按照正常的batch进行索引
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index   # (1,)
            # 根据mask进行box选择
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds  # 预测box的原始数据
            # 如果类别预测是tensor不是list
            if not isinstance(batch_dict['batch_cls_preds'], list):
                # 直接根据mask进行筛选 （B, num_boxes, num_classes | 1)
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]  # (num_boxes, num_classes | 1)

                src_cls_preds = cls_preds  # 预测类别的原始数据 (num_boxes, num_classes)
                assert cls_preds.shape[1] in [1, self.num_class]
                # 如果有正则化需求，则调用sigmod函数对类别分数归一化
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]   # list里的元素还是list
                src_cls_preds = cls_preds  # 预测类别的原始数据
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]
            # 总结：原始的输入数据存在两种形式，一种是batch形式，另一种是index加上连续box的形式
            #      同时输入的数据也有tensor和list两种类型，分开进行处理
            #      获取source的box和class预测结果

            # 2.对该帧点云进行NMS
            # 如果是多类别的NMS
            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]  # 将类别预测的数组打包成list ([arrray1, array2, ...])
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]  # [tensor([0, 1, 2])]
                else:  # 这里之所以直接取值，是因为在上面已经进行处理组成了list
                    multihead_label_mapping = batch_dict['multihead_label_mapping']  # [(num_class1), (num_class2), ...]
                # 总结: 上面的判断和数据组装是为了下面的for循环和zip组装做准备

                cur_start_idx = 0  # 起始id
                pred_scores, pred_labels, pred_boxes = [], [], []
                # zip函数取出对应的结果cur_cls_preds:(num_boxes, num_classes), cur_label_mapping:(num_class)
                # 尽管这里是for循环，但是应该只进行一次for循环，因为一次处理一帧点云数据，不知道理解的是否正确？
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    # 类别数量和标签映射相等
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    # 获取预测box
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    # 调用multi_classes_nms函数进行NMS
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)   # 预测分数
                    pred_labels.append(cur_pred_labels)  # 预测标签
                    pred_boxes.append(cur_pred_boxes)  # 预测box
                    cur_start_idx += cur_cls_preds.shape[0]  # 更新起始索引点
                # 在第0维度上拼接，形成最终预测结果
                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            # 单类别NMS
            else:
                # torch.max()函数的第一个返回值是每行的最大值，第二个返回值是每行最大值的索引
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                # 如果batch_dict没有class_labels
                if batch_dict.get('has_class_labels', False):
                    # 查询是否存在roi_labels，如果存在，则label的键用roi_labels，否则用batch_pred_labels
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]  # 根据index选择预测标签
                else:
                    label_preds = label_preds + 1   # 如果存在类别标签则预测索引+1
                # 调用gpu函数进行nms，返回的是被选择的索引和索引分数
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)   # 预测值的最大值
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores  # 预测分数
                final_labels = label_preds[selected]  # 预测类别
                final_boxes = box_preds[selected]   # 预测box
            # 3. 计算召回率，且recall_dict是其中的参数，循环调用更新
            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST  # RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)  # 字典列表
        # 4. 返回该batch的预测结果和recall
        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        """
                计算召回率
                Args:
                    box_preds:预测box的原始数据（进行NMS之前的）
                    recall_dict:召回率字典（随着预测帧数更新）
                    batch_index:预测数据索引
                    data_dict: 原始预测数据字典
                    thresh_list:阈值列表[0.3, 0.5, 0.7]
                """
        # 如果data_dict中不存在'gt_boxes',无法计算recall,则保持原值并直接返回
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False):
        """
              从文件中加载权重参数
              Args:
                  filename:文件名
                  logger: 日志
                  to_cpu: 是否加载到cpu
              1.torch.load(filename, map_location=loc_type)
              2.self.load_state_dict(state_dict)
              """
        # 如果文件名不是文件，报异常
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)   # 1.调用torch的load函数加载权重文件
        model_state_disk = checkpoint['model_state']  # 在checkpoint中获取model_state

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)
        # 更新各某块权重
        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)
        # 记录未被更新的权重
        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
