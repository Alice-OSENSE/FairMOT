python3 ../demo.py mot \
--load_model "/home/osense-office/Documents/ext_repo/FairMOT/models/fairmot_dla34.pth" \
--conf_thres 0.1 \
--min-box-area 20 \
--track_buffer 90 \
--input-video "/home/osense-office/Documents/sample_video/without_ball_clip.mp4" \
--output-root "/home/osense-office/Documents/ext_repo/FairMOT/results" \
--output-format video