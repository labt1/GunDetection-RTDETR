#ifndef _POST_PROCESS_HPP_CUDA_
#define _POST_PROCESS_HPP_CUDA_

#include <iostream>
#include <cuda_runtime.h>
#include "common/cuda_utils.hpp"

#define BLOCK_SIZE 32

namespace ai
{
    namespace postprocess
    {
        // Generalmente se usa para analizar yolov3/v5/v7/yolox. Si tiene otro posprocesamiento de modelo de tarea que requiere aceleración CUDA, también puede escribirlo aquí.
        // El número máximo predeterminado de cuadros de detección para una imagen es 1024, que se puede cambiar pasando parámetros o modificando directamente los parámetros predeterminados.
        void decode_detect_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                          float confidence_threshold, float *invert_affine_matrix,
                                          float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT, cudaStream_t stream);

        // análisis de posprocesamiento rtdetr
        void decode_detect_rtdetr_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                                 float confidence_threshold, int scale_expand, float *parray, int MAX_IMAGE_BOXES,
                                                 int NUM_BOX_ELEMENT, cudaStream_t stream);
    }
}
#endif // _POST_PROCESS_HPP_CUDA_