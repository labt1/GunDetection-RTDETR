#include "common/arg_parsing.hpp"
#include "common/cv_cpp_utils.hpp"
#include "rtdetr_detect.hpp"

void trt_cuda_video_inference(ai::arg_parsing::Settings *s){

    tensorrt_infer::rtdetr_cuda::RTDETRDetect rtdetr_obj;
    rtdetr_obj.initParameters(s->model_path, s->score_thr);
    
    ai::cvUtil::Image input;
    ai::cvUtil::BoxArray output;
    ai::utils::Timer timer;

    float latency;
    float avg_latency = 0;
    int frames_cont = 0;

    cv::Mat frame;
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    cv::VideoCapture cap(s->video_path);
    if (!cap.isOpened())
    {
        std::cout << "!!! Failed to open file: " << "../input/test.mp4" << std::endl;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::Size frameSize = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
                                  (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));


    cv::Mat mask;
    cv::Mat blurredFrame;

    cv::VideoWriter video("result/outvideo.avi", cv::VideoWriter::fourcc('M','J','P','G'), fps, frameSize);
    int count = 0;

    for(;;){
        if (!cap.read(frame))             
            break;

        mask = cv::Mat::ones(frame.size(), CV_8UC1) * 255;
        
        timer.start();
        input = ai::cvUtil::cvimg_trans_func(frame);
        output = rtdetr_obj.forward(input);

        //cv::putText(frame, "FPS: " + std::to_string(1000/latency), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        for (auto &obj : output)
            ai::cvUtil::draw_one_image_rectangle_2(frame, output, s->classlabels_2, 0.58f);

        latency = timer.stop("Timer", 1, false);
        avg_latency += latency;

        frames_cont++;

        video.write(frame);
        cv::imshow("Camera", frame);
        
        char key = cv::waitKey(10);
        if (key == 27) // ESC
            break;
    }

    cap.release();
    video.release();

    cv::destroyAllWindows();
}

