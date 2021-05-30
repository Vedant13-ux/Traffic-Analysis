import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
# from tracker import *
from model.tracker import *

tracker = EuclideanDistTracker()
tracker_stopped_vehichles=EuclideanDistTracker()


def vehicle_detection(path):

    # Object Detection
    video_stream = cv2.VideoCapture(path)
    fps=video_stream.get(cv2.CAP_PROP_FPS)
    # Randomly select 30 frames
    frameIds = video_stream.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=30)

    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        video_stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = video_stream.read()
        frames.append(frame)
        
    video_stream.release()

    # Calculate the median along the time axis
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

    sample_frame=frames[0]
    
    grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

    graySample=cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)

   
    dframe = cv2.absdiff(graySample, grayMedianFrame)
    blurred = cv2.GaussianBlur(dframe, (11,11), 0)
    ret, tframe= cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    (cnts, _) = cv2.findContours(tframe.copy(), cv2.RETR_EXTERNAL, 
                                cv2 .CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        if y > 200:  #Disregard item that are the top of the picture
            cv2.rectangle(sample_frame,(x,y),(x+w,y+h),(0,255,0),2)

    #Video Processing
    writer = cv2.VideoWriter("output.mp4", 
                            cv2.VideoWriter_fourcc('V','P','8','0'), 20,(640,480))

    #Create a new video stream and get total frame count
    video_stream = cv2.VideoCapture(path)
    total_frames=video_stream.get(cv2.CAP_PROP_FRAME_COUNT)

    frameCnt=0
    left=[]
    right=[]
    left_density=0
    right_density=0

    left_average=[]
    right_average=[]
    passing_junction=[]
    vehicles_every_second=[]
    while(frameCnt < total_frames-1):

        frameCnt+=1
        if(frameCnt%fps==0):
            left_density=math.ceil(np.mean(left)) 
            right_density=math.ceil(np.mean(right))
            left_average.append(left_density)
            right_average.append(right_density)
            vehicles_every_second.append(left_density+right_density)
            left=[]
            right=[]    

        ret, frame = video_stream.read()

        if(frameCnt-1==0):
            frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_copy2= cv2.GaussianBlur(frame_copy, (11,11), 0)
            #Edges of Road
            edges = cv2.Canny(frame_copy2, 50, 200)
            # #ApplyHoughlines
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold = 100, minLineLength=500, maxLineGap=300)
        
            x1_arr=[]
            x2_arr=[]
            y1_arr=[]
            y2_arr=[]

            for line in lines:
                x1, y1, x2, y2 = line[0]
                if (((x1 > 400 and x1<800) and (x2>400 and x2<800)) and ( y1>400 or y2>400 )):
                    x1_arr.append(x1)
                    x2_arr.append(x2)
                    y1_arr.append(y1)
                    y2_arr.append(y2)
            
            x1_line=np.mean(x1_arr)
            x2_line=np.mean(x2_arr)
            y1_line=np.mean(y1_arr)
            y2_line=np.mean(y2_arr)




        # Convert current frame to grayscale
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference of current frame and the median frame
        dframe = cv2.absdiff(gframe, grayMedianFrame)

        # Gaussian
        blurred = cv2.GaussianBlur(dframe, (11, 11), 0)

        #Thresholding to binarise
        ret, tframe= cv2.threshold(blurred,0,255,
                                cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #Identifying contours from the threshold
        (cnts, _) = cv2.findContours(tframe.copy(), 
                                    cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)

        #For each contour draw the bounding bos
        left_vehicles=0
        right_vehicles=0
        detections=[]
        stopped_vehicles=[]

        container_counter=0

        passing_junction_counter=0
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            if x>250 and x<1000 and y > 200 and area>10: 
                container_counter=container_counter+1
                if(y<450 and y>350):
                    passing_junction_counter=passing_junction_counter+1

                if area>25:
                    stopped_vehicles.append([x,y,w,h])
                
                if(area>130 and y>400):
                    detections.append([x,y,w,h])

                if(x < (x1_line+x2_line)/2):
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    left_vehicles=left_vehicles+1
                if(x> (x1_line+x2_line)/2):
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                    right_vehicles=right_vehicles+1
                
        left.append(left_vehicles)
        right.append(right_vehicles)
        passing_junction.append(passing_junction_counter)
        
        #Identifying Stopped Vehicles:
        tracker_stopped_vehichles.update(stopped_vehicles)

        cv2.line(frame, (0, 400), (1200, 400), (0, 255, 0), 2)
        cv2.putText(frame,'Density on Left Lane= %d/sec'%left_density ,(200,40),cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame,'Density on Right Lane= %d/sec'%right_density ,(700,40),cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, (0, 0, 255), 2, cv2.LINE_AA)
            
        cv2.putText(frame,'Vehicles in Frame= %d'%container_counter ,(100,100),cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.putText(frame,'Moving Vehicles= %d'% (container_counter-len(tracker_stopped_vehichles.stopped_vehicles)) ,(500,100),cv2.FONT_HERSHEY_SIMPLEX, 
        0.9, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame,'Stopped Vehicles= %d'%len(tracker_stopped_vehichles.stopped_vehicles) ,(900,100),cv2.FONT_HERSHEY_SIMPLEX, 
        0.9, (0, 0, 255), 2, cv2.LINE_AA)
        
        
        if x1_line !=0:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if (((x1 > 400 and x1<800) and (x2>400 and x2<800)) and ( y1>400 or y2>400 )):
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.line(frame, (int(x1_line), int(y1_line)), (int(x2_line), int(y2_line)), (0, 0, 255), 2)
            
        else:
            cv2.line(frame, (600, 0), (600, 800), (255, 0, 0), 2)
        
        # Object Tracking 
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 3)

        writer.write(cv2.resize(frame, (640,480)))

    # Calculating Stop Time
    stop_time_dict=tracker_stopped_vehichles.stop_time
    stop_time=0
    for v in stop_time_dict.values():
        stop_time=stop_time+v
    stop_time=(stop_time/fps)

    print('Vehicle Count = %d'%tracker.id_count )
    # Summary
    left_average_density=math.ceil(np.average(left_average))
    right_average_density=math.ceil(np.average(right_average))
    passing_junction_density=math.ceil(np.average(passing_junction))
    peak_second=vehicles_every_second[np.max(vehicles_every_second)]
    average_stop_time=stop_time/tracker.id_count

    print(left_average_density)
    print(right_average_density)
    print(passing_junction_density)
    print(peak_second)
    print(average_stop_time)

   
    

    #Release video object
    video_stream.release()
    writer.release()

    return (left_average_density, right_average_density,passing_junction_density,peak_second, average_stop_time)

# vehicle_detection('https://github.com/blackandrose/yolo-pyimagesearch-1/raw/master/videos/overpass.mp4')

