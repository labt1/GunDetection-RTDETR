#ifndef UTILS_ARG_PARSING_H_
#define UTILS_ARG_PARSING_H_
#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <getopt.h>

#define RETURN_SUCCESS (0)
#define RETURN_FAIL (-1)
namespace ai
{
    namespace arg_parsing
    {
        struct Settings
        {
            // Parámetros requeridos
            std::string model_path = ""; // Rutal del modelo
            
            // Parámetros opcionales
            std::string image_path = "";   // Ruta de la imágen
            int batch_size = 1;            // Por defecto es 1, si el modelo es dinámico ingresa el tamaño del lote 
            float score_thr = 0.6f;        // Umbral para filtrar los resultados
            int device_id = 0;             // ID del GPU por si se tienen varias GPUs
            int loop_count = 10;           // La cantidad de veces que la tarea de inferencia se ejecuta en un bucle (No se usa)
            int number_of_warmup_runs = 2; // El nímero de inferencias para calentar el nucleo CUDA (No se usa)
            std::string output_dir = "";   // Dirección de salida donde se guardan los resultados
            std::string camera_ip = "";    // Direccion de la camara IP
            std::string video_path = "";    // Direccion de la camara IP

            // Etiquetas
            const std::vector<std::string> classlabels{"Persona", "bicycle", "Auto", "motorcycle", "airplane", "bs", "train", "Camion", "boat", 
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", 
            "bear", "zebra", "giraffe", "backpack", "mbrella", "Bolsa", "tie", "sitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
            "baseball bat", "baseball glove", "skateboard", "srfboard", "tennis racket", "bottle", "wine glass", "cp", "fork", "knife", "spoon", 
            "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "dont", "cake", "chair", "coch", 
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mose", "remote", "keyboard", "cell phone", "microwave", "oven", 
            "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrsh"};

            const std::vector<std::string> classlabels_2{"Pistola"};
        };
        int parseArgs(int argc, char **argv, Settings *s); // Analiza los parámetros ingresados ​​en la línea de comando y los asigna a Configuración
        void printArgs(Settings *s);                       // Imprimir todos los parámetros
    }
}

#endif // UTILS_ARG_PARSING_H_