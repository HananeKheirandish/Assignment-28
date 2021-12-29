import cv2

video_cap = cv2.VideoCapture(0)

filter_result_list = []
line = 1

while True:
    ret, frame = video_cap.read()

    if ret == False:
        break

    row, col, _ = frame.shape

    cv2.line(frame, (0, line+1), (col, line+1), (0, 255, 255), 1)

    filter_result_list.append(frame[line-1, :])
    frame[:line, :] = filter_result_list

    line += 1

    if line == row:
        cv2.imwrite('result4.jpg', frame)
        break

    cv2.imshow('output', frame)
    cv2.waitKey(1)
