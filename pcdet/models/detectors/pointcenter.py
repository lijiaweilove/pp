from .detector3d_template import Detector3DTemplate


class PointCenter(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # demo.py中调用的是models中__init__.py中的build_network(),返回的是该网络的类
        # 这里调用的是Detector3DTemplate中的build_networks(),
        # 差一个s，这里返回的是各个模块的列表
        self.module_list = self.build_networks()

    def forward(self, batch_dict):  # 一个batch的所以数据
        # Detector3DTemplate构造好所有模块
        # 这里根据模型配置文件生成的配置列表逐个调用forward函数
        for cur_module in self.module_list:  # module_list:见骨架网络结构
            batch_dict = cur_module(batch_dict)  # 最后一个模块CenterHead的forward的输出，带多个属性值的一个batch的data
            # print(batch_dict)

        if self.training:  # 如果在训练模式下，则获取loss
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:  # 在测试模式下，对预测结果进行后处理
            # pred_dicts:预测结果
            # record_dict = {
            #     'pred_boxes': final_boxes,
            #     'pred_scores': final_scores,
            #     'pred_labels': final_labels
            # }
            # recall_dicts:根据全部训练数据得到的召回率
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }  # 此时tb_dict中包含loss_rpn, cls_loss, reg_loss和rpn_loss

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
