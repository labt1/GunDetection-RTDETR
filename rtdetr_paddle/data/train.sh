export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml --eval \
    -r output/rtdetr_r50vd_6x_coco/21.pdparams --use_vdl=True