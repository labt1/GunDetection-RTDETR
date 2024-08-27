python tools/eval.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
              -o weights=output/rtdetr_r18vd_6x_gun/best_model.pdparams

python f1_score.py