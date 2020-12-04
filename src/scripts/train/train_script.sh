python3 ../../train.py mot \
--exp_id mix_dla34_reid \
--load_model '/home/osense-office/Desktop/model_last.pth' \
--data_cfg '../../lib/cfg/data_basketball.json' \
--reid_dim 256 \
--gpus '0,1' \
--batch_size 4 \
--save_dir '' \
--lr 1e-5