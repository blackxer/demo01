import numpy as np
import cv2 as cv
import argparse
import os
from time import time
import datetime
import multiprocessing

SAVE_VIDEO_NAME = {
    # "/media/zw/DL/ly/workspace/project07/demo01/data/2/2019-11-08 11:27:22.avi": "1"
    "/media/zw/DL/ly/workspace/project07/demo01/data/2/2019-11-08 13:17:27.avi": "1",
    # "/media/zw/DL/ly/workspace/project07/demo01/video_out.avi": "2",
}

ALGORITHMS_TO_EVALUATE = [
    (cv.bgsegm.createBackgroundSubtractorMOG, 'MOG', {}),
    (cv.createBackgroundSubtractorMOG2, 'MOG2', {'detectShadows': False}),
    (cv.bgsegm.createBackgroundSubtractorGMG, 'GMG', {}),
    (cv.bgsegm.createBackgroundSubtractorCNT, 'CNT', {'useHistory': True}),
    (cv.bgsegm.createBackgroundSubtractorLSBP, 'LSBP-vanilla', {'nSamples': 20, 'LSBPRadius': 4, 'Tlower': 2.0, 'Tupper': 200.0, 'Tinc': 1.0, 'Tdec': 0.05, 'Rscale': 5.0, 'Rincdec': 0.05, 'LSBPthreshold': 8}),
    (cv.bgsegm.createBackgroundSubtractorLSBP, 'LSBP-speed', {'nSamples': 10, 'LSBPRadius': 16, 'Tlower': 2.0, 'Tupper': 32.0, 'Tinc': 1.0, 'Tdec': 0.05, 'Rscale': 10.0, 'Rincdec': 0.005, 'LSBPthreshold': 8}),
    (cv.bgsegm.createBackgroundSubtractorLSBP, 'LSBP-quality', {'nSamples': 20, 'LSBPRadius': 16, 'Tlower': 2.0, 'Tupper': 32.0, 'Tinc': 1.0, 'Tdec': 0.05, 'Rscale': 10.0, 'Rincdec': 0.005, 'LSBPthreshold': 8}),
    (cv.bgsegm.createBackgroundSubtractorLSBP, 'LSBP-camera-motion-compensation', {'mc': 1}),
    (cv.bgsegm.createBackgroundSubtractorGSOC, 'GSOC', {}),
    (cv.bgsegm.createBackgroundSubtractorGSOC, 'GSOC-camera-motion-compensation', {'mc': 1})
]
KERNEL = []

def analyse(video_source):
    argparser = argparse.ArgumentParser(description='Vizualization of the LSBP/GSOC background subtraction algorithm.')
    argparser.add_argument('-a', '--algorithm', help='Test particular algorithm instead of all.', default='MOG2')
    args = argparser.parse_args()

    VIDEO_NAME = video_source #"rtsp://admin:hk123456@112.250.110.15:554/Streaming/Channels/101?transportmode=unicast"
    cap = cv.VideoCapture(VIDEO_NAME)
    assert cap.isOpened(), "receieve stream failed"
    fps = cap.get(cv.CAP_PROP_FPS)
    pauseInput = int(1000 // fps)
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)*2), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    # video_writer = cv.VideoWriter("data/{}/{}.avi".format(SAVE_VIDEO_NAME[VIDEO_NAME],datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')), cv.VideoWriter_fourcc(*'MP42'), fps, size)
    video_writer = cv.VideoWriter("detect_out.avi", cv.VideoWriter_fourcc(*'MP42'), fps, size)

    if args.algorithm is not None:
        global ALGORITHMS_TO_EVALUATE
        ALGORITHMS_TO_EVALUATE = filter(lambda a: a[1].lower() == args.algorithm.lower(), ALGORITHMS_TO_EVALUATE)
    areas = []
    for algo, algo_name, algo_arguments in ALGORITHMS_TO_EVALUATE:
        print('Algorithm name: %s' % algo_name)
        bgs = algo(**algo_arguments)
        i=0
        ts = time()
        while True:
            success, frame = cap.read()
            ori_frame = frame.copy()
            if cv.waitKey(1) == 27:
                break

            if not success:
                break
            i+=1
            # cv.imshow('Frame', frame)

            # img_gray = cv.cvtColor(f[i], cv.COLOR_BGR2GRAY)
            # img = cv.morphologyEx(img_gray, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)))
            # img = cv.GaussianBlur(f[i], (5,5),0) # 高斯模糊
            img = cv.blur(frame, (7, 7))  # 均值模糊
            img = cv.medianBlur(img, 7) # 中值模糊
            # img = cv.bilateralFilter(f[i],9,75,75)

            # cv.imshow('tophat',img)
            # cv.moveWindow('tophat',900,100)
            mask = bgs.apply(img,1)
            # cv.namedWindow("mask0", cv.WINDOW_NORMAL)
            # cv.imshow("mask0", mask)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)))
            # mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)))
            # cv.namedWindow("mask1", cv.WINDOW_NORMAL)
            # cv.imshow("mask1", mask)
            contours, hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv.contourArea(cnt)
                print(area)
                if area>0:
                    x,y,w,h = cv.boundingRect(cnt)
                    cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
                    # cv.drawContours(f[i], [cnt], 0, (0,255,0),3)
                    # cv.imshow('area', f[i])
                    # cv.moveWindow('area',100,400)
                # else:
                #     cv.imshow('area', f[i])
                #     cv.moveWindow('area', 100, 400)

            # bg = bgs.getBackgroundImage()
            # cv.imshow('BG', bg)
            # cv.imshow('Output mask', mask)
            # cv.moveWindow('Output mask',500,400)
            # mask = np.expand_dims(mask, axis=2)
            # mask = np.concatenate((mask, mask, mask), axis=-1)
            # result = cv.addWeighted(frame, 0.7, mask, 0.3, 0)
            # cv.namedWindow("result-{}".format(SAVE_VIDEO_NAME[VIDEO_NAME]), cv.WINDOW_NORMAL)
            result = cv.hconcat([ori_frame, frame])
            cv.imshow("result-{}".format(SAVE_VIDEO_NAME[VIDEO_NAME]), result)
            video_writer.write(result)

    video_writer.release()
    cap.release()
    # cv.destroyAllWindows()
    print("over")


if __name__ == '__main__':
    rtsp_source = list(SAVE_VIDEO_NAME.keys())

    m1 = multiprocessing.Process(target=analyse, args=(rtsp_source[0],))
    m1.start()
    m1.join()

    # m2 = multiprocessing.Process(target=analyse, args=(rtsp_source[1],))
    # m2.start()
    # m2.join()