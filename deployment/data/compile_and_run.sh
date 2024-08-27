# rm -rf build && mkdir -p build && cd build && cmake ..  && make -j48 && cd ..
mkdir -p build && cd build && cmake ..  && make -j48 && cd ..
cd output

# Para una imagen (test) -c 100 -w 10 -b 1
: '
./infer -f ../models/rtdetr_r18_B.trt \
        -i ../input/cap1.jpg \
        -o result \
        -c 1 -w 1 -b 1 -s 0.30
'

#: '
./infer -f ../models/rtdetr_r18_B.trt \
        -s 0.40f -b 1\
        -v "../input/test1.mp4"
#'


: '
./infer -f ../models/rtdetr_r18_B.trt \
        -s 0.86f -b 1\
        -p "rtsp://admin:unsa2024@192.168.0.217:554/cam/realmonitor?channel=1&subtype=0"
'


cd ..