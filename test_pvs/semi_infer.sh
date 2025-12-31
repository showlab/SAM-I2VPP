################## SAM-I2VPP ##################

# setup prompt
#input="3c"
#input="bb"
input="gm"

# setup path
method="i2vpp"
ckpt="sam-i2vpp_8gpu"
yaml="i2vpp-infer.yaml"
save_dir_name="sam-i2vpp_8gpu"

# run inference
python inference_Semi_SAV_mgpu.py \
--input ${input} \
--method ${method} \
--ckpt ${ckpt} \
--yaml ${yaml} \
--save_dir_name ${save_dir_name}
