import numpy as np
import cv2 as cv

video_file = '/chessboard.mp4'
K = np.array([[350, 0, 350],
              [0, 250, 250],
              [0, 0, 1]]) 
dist_coeff = np.array([-0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194])


video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

fourcc = cv.VideoWriter_fourcc(*'XVID')  

show_rectify = True
map1, map2 = None, None
while True:
    valid, img = video.read()
    if not valid:
        break

    info = "Original"
    if show_rectify:
        if map1 is None or map2 is None:
            map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, None, (img.shape[1], img.shape[0]), cv.CV_32FC1)
        img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
        info = "Rectified"
    cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    out = cv.VideoWriter('output.mp4', fourcc, 20.0, (img.shape[1], img.shape[0]))

    out.write(img)

    cv.imshow("Geometric Distortion Correction", img)
    key = cv.waitKey(10)
    if key == ord(' '):     # Space: Pause
        key = cv.waitKey()
    if key == 27:           # ESC: Exit
        break
    elif key == ord('\t'):  # Tab: Toggle the mode
        show_rectify = not show_rectify


video.release()
out.release()
cv.destroyAllWindows()
