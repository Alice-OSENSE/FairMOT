python ../../train.py mot \
--exp_id mix_dla34_reid \
--load_model 'Users\chengyu\Desktop\Alice\weights' \
--data_cfg '../../lib/cfg/data_basketball.json' \
--reid_dim 256 \
--gpus '0,1' \
--batch_size 4 \
--save_dir 'Users\chengyu\Desktop\Alice\exp' \
--lr 1e-5