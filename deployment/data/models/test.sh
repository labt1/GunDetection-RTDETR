trtexec --loadEngine=rtdetr_r18_B.trt \
        --shapes=image:1x3x640x640 \
        --avgRuns=1000 \
        --fp16 \

#trtexec --loadEngine=def.trt \
#        --shapes=image:1x3x640x640 \
#        --avgRuns=1000 \
#        --fp16