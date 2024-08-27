# Si el modelo tiene una entrada dinamica incluye el siguiente flag donde se especifica el tamaño de la entrada 
# --shapes=images:1x3x640x640 
# Si el modelo es estatico omite para el tipo de dato --fp16 --int8 , etc

#--shapes=image:1x3x640x640
#--minShapes=image:1x3x640x640 \
#--maxShapes=image:16x3x640x640 \
#--optShapes=image:4x3x640x640 \
# No usar --useSpinWait ya que aumenta la latencia

#trtexec --onnx=model.onnx --saveEngine=rtdetr_r50_pytorch_fp32.trt --memPoolSize=workspace:4096 --avgRuns=100
trtexec --onnx=rtdetr_r18_B.onnx --saveEngine=rtdetr_r18_B.trt \
        --shapes=image:1x3x640x640 \
        --avgRuns=100 --fp16 --workspace=20480 --useCudaGraph