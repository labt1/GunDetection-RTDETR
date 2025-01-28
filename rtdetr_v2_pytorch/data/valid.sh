CUDA_VISIBLE_DEVICES=0 torchrun --master-port=8989 tools/train.py \
    -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_bifpn.yml \
    -r output_bifpn_MGD/best.pth \
    -d cuda \
    --test-only