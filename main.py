#Modules: computer vision (cv2), imutils to resize & position frames, & datetime
import cv2, imutils, datetime

#Video Capture variable/object, 0 is default webcam location
video_capture = cv2.VideoCapture(0)

#Object detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=20)

#The initial frame needs to be static to capture the background, as this is necessary to make comparisons with current frames and determine whether there is a difference.
initial_frame = None 

#Program runs in a while loop until it is quit. Hence why all code below is indented
while True:
    #Variable frame uses function read frames, used to spot difference between frames
    frame = video_capture.read()[1] #required two variables name but i just had one and set index to [1] instead
   

    #Frames converted into greyscale, necessary for threshold
    greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Blur the image to remove noise and increase program's ability to calculate difference in frame (spot movement)
    gaussian_frame = cv2.GaussianBlur(greyscale, (21,21), 0) 
    blur_frame = cv2.blur(gaussian_frame, (5,5))
    greyscale_frame = blur_frame


    #threshold frame as object detecor live frame. Named threshold because of threshold attribute on line 8 
    threshold_frame = object_detector.apply(frame)
    #Added another threshold function to prevent detection of shadows etc.
    _, threshold_frame = cv2.threshold(threshold_frame, 254, 255, cv2.THRESH_BINARY)


    #Condition causes background to become grey, one it reaches continue it reiterates through the loop but this time condition isn't met as initial_image = greyscale, and not 'None'
    if initial_frame is None:
        initial_frame = greyscale
        continue

    #Initial frame acts as background. Delta frame compares the initial frame with the current frame, to spot differences caused by motion/movement
    delta_frame = cv2.absdiff(initial_frame, greyscale)

    #Finding the boundaries of a moving objects aka Contours
    (contours,_) = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cntr in contours:
        #Contour needs an area above 1000 to be detected otherwise loop will reloop/continue
        if cv2.contourArea(cntr) < 1000:
            continue
        else:
        #If contour greater than 1K, then it will be drawn around object. drawContours() is more accurate than rectangle
            cv2.drawContours(frame, [cntr], -1, (0, 255, 0), 2)

        
        #Texts on main frame that display when motion is detected
        motion_detection = cv2.putText(frame, "Motion Detected", 
        (10,20), cv2.FONT_HERSHEY_TRIPLEX , 0.60, (0, 5, 255), 2)
        #Displays Date and time when object is in view
        motion_detection_time = cv2.putText(frame, datetime.datetime.now().strftime("%m/%d/%y, %H:%M:%S"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX , 0.525, (0, 0, 255),1)
    
    
    #Resizing frames
    greyscale_frame = imutils.resize(greyscale_frame, width=600)
    delta_frame = imutils.resize(delta_frame, width=600)
    threshold_frame = imutils.resize(threshold_frame, width=600)
    frame = imutils.resize(frame, width=600)


   #imshow function to display the various frames
    cv2.imshow("Live Footage", greyscale_frame)
    cv2.imshow("Color Frame", frame)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", threshold_frame)
    

    #Positioning all 4 windows
    cv2.moveWindow("Live Footage", 325, 30)
    cv2.moveWindow("Color Frame", 980, 30)
    cv2.moveWindow("Delta Frame", 325, 530)
    cv2.moveWindow("Threshold Frame", 980, 530)
   

    #Quiting the program. Once q is pressed, program waits a second before breaking the while loop
    key = cv2.waitKey(1) & 0xFF
    quit = "q"
    if key == ord(quit.lower()):
        break

#Release video_capture object from lock, turning of the camera.
video_capture.release()
#Removes all frames displaying live video footage
cv2.destroyAllWindows

