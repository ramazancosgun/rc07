#step motor kontrol ve servo motor
import cv2
import numpy as np
# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import time
import argparse
import imutils
import RPi.GPIO as GPIO
GPIO.setwarnings(False)
cap = cv2.VideoCapture(0)
GPIO.setmode(GPIO.BCM)
GPIO.setup(25,GPIO.OUT)
GPIO.setup(5,GPIO.OUT)
GPIO.setup(24,GPIO.OUT)
GPIO.setup(23,GPIO.OUT)
#step bölümü
enable_pin = 18
coil_A_1_pin = 4
coil_A_2_pin = 17
coil_B_1_pin = 27
coil_B_2_pin = 22
GPIO.setup(enable_pin, GPIO.OUT)
GPIO.setup(4, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)
GPIO.setup(27, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)
GPIO.output(enable_pin, 1)


servo1 = GPIO.PWM(25,180)
servo2 = GPIO.PWM(5,180)
servo3 = GPIO.PWM(24,180)
servo4 = GPIO.PWM(23,180)

servo1.start(22.5)
servo2.start(22.5)
servo3.start(11)
servo4.start(11)

time.sleep(1)
GPIO.cleanup()

def forward(delay, steps):
    
    for i in range(0, steps):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(enable_pin, GPIO.OUT)
        GPIO.setup(4, GPIO.OUT)
        GPIO.setup(17, GPIO.OUT)
        GPIO.setup(27, GPIO.OUT)
        GPIO.setup(22, GPIO.OUT)
        GPIO.output(enable_pin, 1)
        setStep(1, 0, 1, 0)
        time.sleep(delay)
        setStep(0, 1, 1, 0)
        time.sleep(delay)
        setStep(0, 1, 0, 1)
        time.sleep(delay)
        setStep(1, 0, 0, 1)
        time.sleep(delay)
def backwards(delay, steps):
    for i in range(0, steps):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(enable_pin, GPIO.OUT)
        GPIO.setup(4, GPIO.OUT)
        GPIO.setup(17, GPIO.OUT)
        GPIO.setup(27, GPIO.OUT)
        GPIO.setup(22, GPIO.OUT)
        GPIO.output(enable_pin, 1)
        setStep(1, 0, 0, 1)
        time.sleep(delay)
        setStep(0, 1, 0, 1)
        time.sleep(delay)
        setStep(0, 1, 1, 0)
        time.sleep(delay)
        setStep(1, 0, 1, 0)
        time.sleep(delay)
def setStep(w1, w2, w3, w4):
    GPIO.output(4, w1)
    GPIO.output(17, w2)
    GPIO.output(27, w3)
    GPIO.output(22, w4)

GPIO.cleanup()

def go_home(timerhome):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(25,GPIO.OUT)
    servo1.start(5)
    time.sleep(timerhome)
    GPIO.cleanup()

def elma_al(timer0):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(25,GPIO.OUT)
    servo1.start(50)
    time.sleep(timer0)
    GPIO.cleanup()
    
def saglama_git(timer1):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(25,GPIO.OUT)
    servo1.start(30)
    time.sleep(timer1)
    GPIO.cleanup() 
 
def servo1_run(timer1, angle):
    print("Servo1")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(25,GPIO.OUT)
    servo1.start(angle)
    time.sleep(timer1)
    GPIO.cleanup()

def servo2_run(timer1, angle):
    print("Servo2")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(5,GPIO.OUT)
    servo2.start(angle)
    time.sleep(timer1)
    GPIO.cleanup()
    
def servo3_run(timer1, angle):
    print("Servo3")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(24,GPIO.OUT)
    servo3.start(angle)
    time.sleep(timer1)
    GPIO.cleanup()
    
def servo4_run(timer1, angle):
    print("Servo4")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(23,GPIO.OUT)
    servo4.start(angle)
    time.sleep(timer1)
    GPIO.cleanup()    
    

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def curuge_git(timer2):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(25,GPIO.OUT)
    servo1.start(22.5)
    time.sleep(timer2)
    GPIO.cleanup()
try:
    while(1):
        forward(0.003, 200)
        cap.set(3,640)
        cap.set(4,480)
        GPIO.setmode(GPIO.BCM)

        a=1
        b=1
        # Take each frame
        _, frame = cap.read()
        _, frame1 = cap.read()
        background = cv2.flip(frame, 1)
        # load the image, convert it to grayscale, and blur it slightly
        image = frame
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # perform edge detection, then perform a dilation + erosion to
        # close gaps in between object edges
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
         
        # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
         
        # sort the contours from left-to-right and initialize the
        # 'pixels per metric' calibration variable
        if len(cnts)>0:
            (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None
                
                
        # loop over the contours individually
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 100:
                continue
         
            # compute the rotated bounding box of the contour
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
         
            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
         
            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
         
            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
         
            # draw the midpoints on the image
            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
         
            # draw lines between the midpoints
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                (255, 0, 255), 2)
            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            if pixelsPerMetric is None:
                pixelsPerMetric = dB / 0.955  
            # compute the size of the object
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric
            print("Inc: dimA: " + "{:.1f}".format(dimA))
            print("Inc: dimB: " + str(dimB))
            # draw the object sizes on the image
            cv2.putText(orig, "{:.1f}in".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
            cv2.putText(orig, "{:.1f}in".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
         
            # show the output image
            cv2.imshow("Image", orig)
            
         
            # if the pixels per metric has not been initialized, then
            # compute it as the ratio of pixels to supplied metric
            # (in this case, inches)


 
    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)        

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        cv2.imshow("screen1", frame)
        cv2.imshow("screen2", frame1)

        # define range of blue color in HSV
        lower_crk = np.array([90,50,50])
        upper_crk = np.array([130,255,255])

    
        lower_saglam = np.array([0,50,100])
        upper_saglam = np.array([10,255,255])


        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_crk, upper_crk)
        mask = cv2.erode(mask,None,iterations=2)
        mask = cv2.dilate(mask,None,iterations=2)
    
        mask1= cv2.inRange(hsv1, lower_saglam, upper_saglam)
        mask1 = cv2.erode(mask1,None,iterations=2)
        mask1 = cv2.dilate(mask1,None,iterations=2)
    


        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
        mask1 = cv2.GaussianBlur(mask1, (3, 3), 0)
        cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    

        if len(cnts)>0:
            
            print("SIYAH TESPIT EDILDI")

            while(a==1):
                print("SERVO FUNCTIONS FOR BLACK")

                forward(0.003, 5)
                time.sleep(1) 
                servo1_run(1, 40)
                time.sleep(1) 
                servo2_run(1, 40)
                time.sleep(1) 
                servo3_run(1, 40)
                time.sleep(1) 
                servo4_run(1, 40)
                #go_home(1)
                time.sleep(1)
                a=0
            
                
        if len(cnts1)>0:
            
            print("RENK KIRMIZI")
            while(b==1):
                print("SERVO RUN FUNCTIONS")
                forward(0.003, 5)
                time.sleep(1)
                #saglama_git(1)
                time.sleep(1) 
                servo1_run(1, 2.5)
                time.sleep(1) 
                servo2_run(1, 2.5)
                time.sleep(1) 
                servo3_run(1, 2.5)
                time.sleep(1) 
                servo4_run(1, 2.5)
                #go_home(1)
                time.sleep(1)
                b=0
         

        #cv2.imshow('frame',frame)
        #cv2.imshow('mask',mask)
        #cv2.imshow('res',res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
  
except KeyboardInterrupt:
    GPIO.cleanup()  

       

cv2.destroyAllWindows()

