import cv2
import numpy as np

def cartoonize_image(img, k=9, edge_thresh=100):
    # 컬러 양자화를 위한 k-평균 클러스터링
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(img.shape)

    # Canny 에지 검출
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray, edge_thresh, edge_thresh*2)

    # 에지 강조
    edges = cv2.bitwise_not(edges)
    cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)

    return cartoon

# 비디오 파일을 읽기 위한 준비
cap = cv2.VideoCapture('video.avi')  # 'video.mp4'는 원하는 비디오 파일 경로로 변경해주세요.

# 비디오 저장을 위한 준비
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('cartoon_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cartoon_frame = cartoonize_image(frame, k=9, edge_thresh=100)
    
    # 변환된 프레임을 비디오 파일로 저장
    out.write(cartoon_frame)
    
    cv2.imshow('Cartoonized Video', cartoon_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 작업이 끝나면 해제
cap.release()
out.release()
cv2.destroyAllWindows()
