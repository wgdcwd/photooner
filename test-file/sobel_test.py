import cv2
import numpy as np

# 비디오를 불러오기 위한 준비
cap = cv2.VideoCapture('video.mp4')  # 'video.mp4'를 원하는 비디오 파일 경로로 변경해주세요.

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 중간값 블러를 적용하여 노이즈를 줄임
    gray = cv2.medianBlur(gray, 5)

    # Sobel 연산자를 사용하여 x, y 방향의 에지 검출
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    # Sobel 마그니튜드를 계산
    sobelMagnitude = np.sqrt(sobelx**2 + sobely**2)
    # Normalize and convert to uint8
    sobelMagnitude = cv2.convertScaleAbs(sobelMagnitude)

    # 임계값을 적용하여 에지를 이진화
    _, edges = cv2.threshold(sobelMagnitude, 100, 255, cv2.THRESH_BINARY)

    # 바이레터럴 필터를 적용하여 컬러 이미지를 준비
    color = cv2.bilateralFilter(frame, 9, 300, 300)

    # 컬러 이미지와 에지 마스크를 결합
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    # 결과를 표시
    cv2.imshow('Cartoon', cartoon)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
