# Weight V3  Datset V1

python detect.py --cfg cfg/polyp_train_yolov4-pacspcbam_v2.cfg --names data/polyp.names --weights weights/best_yolov4-pacspcbam_v2.pt --source  /home/sharib/Desktop/5teams_EndoCV/endocv2021-test-noCopyAllowed-v3/EndoCV_DATA1/ --img-size 640 --conf-thres 0.001 --iou-thres 0.01 --device 0 --save-txt

python detect.py --cfg cfg/polyp_train_yolov4-pacspcbam_v2.cfg --names data/polyp.names --weights weights/best_yolov4-pacspcbam_v2.pt --source  /home/sharib/Desktop/5teams_EndoCV/endocv2021-test-noCopyAllowed-v3/EndoCV_DATA2/ --img-size 640 --conf-thres 0.001 --iou-thres 0.01 --device 0 --save-txt


python detect.py --cfg cfg/polyp_train_yolov4-pacspcbam_v2.cfg --names data/polyp.names --weights weights/best_yolov4-pacspcbam_v2.pt --source  /home/sharib/Desktop/5teams_EndoCV/endocv2021-test-noCopyAllowed-v3/EndoCV_DATA3/ --img-size 640 --conf-thres 0.001 --iou-thres 0.01 --device 0 --save-txt

python detect.py --cfg cfg/polyp_train_yolov4-pacspcbam_v2.cfg --names data/polyp.names --weights weights/best_yolov4-pacspcbam_v2.pt --source  /home/sharib/Desktop/5teams_EndoCV/endocv2021-test-noCopyAllowed-v3/EndoCV_DATA4/ --img-size 640 --conf-thres 0.001 --iou-thres 0.01 --device 0 --save-txt





