import cv2
import numpy as np

def video_to_cartoon(video_path):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    
    # 출력 파일 설정
    output_file = 'cartoon_video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 만화 스타일 변환 프로세스 시작
        # 1. 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 2. 미디언 블러로 노이즈 감소
        gray = cv2.medianBlur(gray, 15)
        # 3. 적응형 임계값을 통한 에지 검출
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        # 4. 양방향 필터로 색상 개선
        color = cv2.bilateralFilter(frame, 9, 275, 275)
        # 5. 색상 이미지와 에지 마스크 결합
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        # 변환된 프레임을 출력 파일에 작성
        out.write(cartoon)

    # 작업 완료 후 자원 해제
    cap.release()
    out.release()

# 여기에 비디오 파일의 경로를 입력하세요.
video_path = 'input_video.mp4'
video_to_cartoon(video_path)