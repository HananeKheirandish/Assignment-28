import cv2
from imutils.perspective import four_point_transform

video_cap = cv2.VideoCapture(0)

while True:
    ret, frame = video_cap.read()

    if ret == False:
        break

    row, col, _ = frame.shape

    img = frame
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 3)

    thresh = cv2.adaptiveThreshold(img_blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]
    contours = sorted(contours, key= cv2.contourArea, reverse = True)

    sudoku_contour = None

    for contour in contours:
        epsilon = 0.12 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            sudoku_contour = approx
            break
            
    if sudoku_contour is None:
        print('not found.')
    else:
        result = cv2.drawContours(img, [sudoku_contour], -1, (0, 255, 0), 4)
        warped = four_point_transform(img, approx.reshape(4,2))

        cv2.imshow('sudoku', result)
        cv2.imwrite('result_cap1.jpg', result)

        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite('result_cap2.jpg', warped)

        if key == ord('q'):
            break