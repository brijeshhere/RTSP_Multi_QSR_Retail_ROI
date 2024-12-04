import cv2
import os

def frame(video_path):
    cap=cv2.VideoCapture(video_path)
    
    frame_num=0
    while True:
        ret,frame=cap.read()
        frame_num+=1
        
        if frame_num%15==0:
            filename="frame_15.jpg"
            full_path=os.path.join(os.path.dirname(video_path),filename)
            frame=cv2.resize(frame,(1920,1080),interpolation=cv2.INTER_AREA)
            cv2.imwrite(full_path,frame)
            print(f"frame 15 saved as {filename}")
            print(frame.shape)
            break
    cap.release()
    cv2.destroyAllWindows()
    return None
        