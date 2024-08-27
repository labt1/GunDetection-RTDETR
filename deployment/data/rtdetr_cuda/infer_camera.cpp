#include "common/arg_parsing.hpp"
#include "common/cv_cpp_utils.hpp"
#include "rtdetr_detect.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "camera/RTSPcam.h"

int trt_cuda_video_stream_inference(ai::arg_parsing::Settings *s)
{
    tensorrt_infer::rtdetr_cuda::RTDETRDetect rtdetr_obj;
    rtdetr_obj.initParameters(s->model_path, s->score_thr);

    tensorrt_infer::rtdetr_cuda::RTDETRDetect rtdetr_obj_coco;
    rtdetr_obj_coco.initParameters("../models/rtdetr_r50_static_coco_fp16.trt", 0.90);

    ai::cvUtil::Image input;
    ai::cvUtil::BoxArray output;
    ai::cvUtil::BoxArray output_coco;
    ai::utils::Timer timer;

    float latency;
    float avg_latency = 0;
    int frames_cont = 0;

    RTSPcam cam;
    cam.Open(s->camera_ip);
    cv::Mat frame;
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    cv::Size frameSize = cv::Size((int)1920,
                                  1080);

    cv::VideoWriter video("result/camara_rec.mp4", cv::VideoWriter::fourcc('m','p','4','v'), 30, frameSize);

    while(true) {
        
        if(!cam.GetLatestFrame(frame)){
            cout << "Capture read error" << endl;
            break;
        }
        
        timer.start();

        input = ai::cvUtil::cvimg_trans_func(frame);
        output = rtdetr_obj.forward(input);
        output_coco = rtdetr_obj_coco.forward(input);

        //cv::putText(frame, "FPS: " + std::to_string(1000/latency), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        ai::cvUtil::draw_one_image_rectangle(frame, output_coco, s->classlabels, 1.0f);
        ai::cvUtil::draw_one_image_rectangle_2(frame, output, s->classlabels_2, 1.0f);

        latency = timer.stop("Timer", 1, false);
        avg_latency += latency;
        frames_cont++;

        video.write(frame);
        cv::imshow("Camera",frame);
        char esc = cv::waitKey(1);
        if(esc == 27) break;
    }

    cam.~RTSPcam();
    video.release();

    std::cout<<"Promedio latencia: "<<avg_latency/frames_cont<<std::endl;
    std::cout<<"Promedio FPS: "<<1000/(avg_latency/frames_cont)<<std::endl;

    std::cout<<"Promedio preprocess: "<<rtdetr_obj.preprocess_time/frames_cont<<std::endl;
    std::cout<<"Promedio forward(): "<<rtdetr_obj.inference_time/frames_cont<<std::endl;
    std::cout<<"Promedio postprocess: "<<rtdetr_obj.postprocess_time/frames_cont<<std::endl;

    cv::destroyAllWindows();
}