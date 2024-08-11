import cv2
import streamlit as st
import os
import ultralytics


model=ultralytics.YOLO('yolov8n.pt')

st.title("Object Tracker")


video_file=st.file_uploader('Upload Video',type=['mp4','avi','mov','mkv'])

if video_file:
    data=video_file.read()
    with open("temp.mp4",'wb') as f:
        f.write(data)


    cap=cv2.VideoCapture('temp.mp4')

    if cap:
        frame_placeholder=st.empty()
        ret=True

        while ret:

            ret,frame=cap.read()

            if ret:

                results=model.track(frame,persist=True)
                out=results[0].plot()
                out=cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
                frame_placeholder.image(out)


        cap.release()
        os.remove('temp.mp4')




