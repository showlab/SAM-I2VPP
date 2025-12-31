################## SAM-I2VPP ##################

# setup prompt
#input="3c"
#input="bb"
input="gm"

# setup path
method="i2vpp"
ckpt="sam-i2vpp_8gpu"
save_dir_name="sam-i2vpp_8gpu"

# setup workers
workers=64

# run evaluation
prediction_name="Semi_SAVTest_${method}_${ckpt}_${input}"
python ../tools/sav_evaluator.py \
--gt_root /workspace/i2vpp/data/sav_test/Annotations_6fps \
--pred_root ./output_semi/${save_dir_name}/${prediction_name} \
--num_processes ${workers}
