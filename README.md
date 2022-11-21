#环境
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c conda-forge
pip install -r  requirements.txt
pip install spconv-cu111
#生成pkl文件
python -m pcdet.datasets.kitti.game_dataset create_game_infos tools/cfgs/dataset_configs/game_dataset.yaml
#训练
cd tools && python train.py --cfg_file cfgs/kitti_models/gamepillar.yaml
#测试
python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --batch_size 4 --ckpt ../output/kitti_models/pointpillar/default/ckpt/checkpoint_epoch_80.pth
#demo可视化
python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml  --data_path ../data/stone/testing/lidar/000001.bin --ckpt ../output/kitti_models/pointpillar/default/ckpt/checkpoint_epoch_80.pth
python demo.py --cfg_file cfgs/kitti_models/pointcenter.yaml  --data_path ../data/stone/testing/lidar/000001.bin --ckpt ../output/kitti_models/pointcenter/default/ckpt/checkpoint_epoch_80.pth
#目前的修改
1.get_infos中将y变为-y
2.打印输出generate_prediction_dicts中预测结果anno
3.
