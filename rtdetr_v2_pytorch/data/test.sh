#CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --test-only

python inference.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_bifpn.yml -r output_bifpn_MGD/best.pth \
        -f test_images/pistol_1013.jpg \
        -d cuda \
        -t 0.50 
