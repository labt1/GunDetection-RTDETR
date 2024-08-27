python tools/x2coco.py \
        --dataset_type voc \
        --voc_anno_dir dataset/gun/val/Annotations/ \
        --voc_anno_list dataset/gun/val/val.txt \
        --voc_label_list dataset/gun/val/label_list.txt \
        --voc_out_name voc_valid.json \
        --output_dir ./dataset/gun_coco/ \