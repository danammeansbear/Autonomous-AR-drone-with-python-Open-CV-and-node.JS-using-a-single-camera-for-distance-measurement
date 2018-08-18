# import the necessary packages
from picamera.array import PiRGBArray     #As there is a resolution problem in raspberry pi, will not be able to capture frames by VideoCapture
from picamera import PiCamera

import time
import cv2
import cv2.cv as cv
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('sachin.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)


def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth
#initialze the know distance from the camera
KNOWN_DISTANCE = 36

#now initialize the know object (we assume a face is 12 inches
KNOWN_WIDTH 11.0

image = cv2.imread('img',img) 
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

for imagePath in sorted(paths.list_images("images")):
# load the image and find the market in the image then
#compute the distance to the marker from the camera
	box = cv2.cv.Box.Points(marker) if imutils.is_cv2() else cv2.boxPoints (marker)
	box = np.int0(box)
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	cv2.putText(image, "%.2fft" % (inches / 12),(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 3)
	cv2.imshow("image", image)




def forward():
	from Naked.toolshed.shell import execute_js

	success = execute_js('forwards.js')
def backward():
	from Naked.toolshed.shell import execute_js

	success = execute_js('backwards.js')
def left():
	from Naked.toolshed.shell import execute_js

	success = execute_js('left.js')
def Right():
	from Naked.toolshed.shell import execute_js

	success = execute_js('right.js')
def Stop():
	from Naked.toolshed.shell import execute_js

	success = execute_js('stop.js')

	
def gotopeople ():
	for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
      #grab the raw NumPy array representing the image, then initialize the timestamp and occupied/unoccupied text
      frame = image.array
      frame=cv2.flip(frame,1)
      global centre_x
      global centre_y
      centre_x=0.
      centre_y=0.
      hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      mask_red=segment_colour(frame)      #masking red the frame
      loct,area=find_blob(mask_red)
      x,y,w,h=loct
	  
	  
	   if (w*h) < 10:
            found=0
      else:
            found=1
            simg2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
            centre_x=x+((w)/2)
            centre_y=y+((h)/2)
            cv2.circle(frame,(int(centre_x),int(centre_y)),3,(0,110,255),-1)
            centre_x-=160
            centre_y-=120
            print 'X:',centre_x, 'Y:',centre_y
      initial=400
      flag=0
	  
	  
	   if(found==0):
            #if the person is not found and the last time it sees the person in which direction, it will start to rotate in that direction
            if flag==0:
                  Right()
                  time.sleep(0.05)
            else:
                  left()
                  time.sleep(0.05)
            stop()
            time.sleep(0.0125)
		elif(found==1):
            if(area<initial):
                  if(distance_to_camera<10):
                        #if Person is too far but it detects something in front of it,then it avoid it and go to people
                        if distanceR>=8:
                              Right()
                              time.sleep(0.00625)
                              stop()
                              time.sleep(0.0125)
                              forward()
                              time.sleep(0.00625)
                              stop()
                              time.sleep(0.0125)
                              #while found==0:
                              left()
                              time.sleep(0.00625)
                        elif distanceL>=8:
                              left()
                              time.sleep(0.00625)
                              stop()
                              time.sleep(0.0125)
                              forward()
                              time.sleep(0.00625)
                              stop()
                              time.sleep(0.0125)
                              Right()
                              time.sleep(0.00625)
                              stop()
                              time.sleep(0.0125)
                        else:
                              stop()
                              time.sleep(0.01)
                  else:
                        #otherwise it move forward
                        forward()
                        time.sleep(0.00625)
            elif(area>=initial):
                  initial2=6700
                  if(area<initial2):
                        if(distance_to_camera>10):
                              #it brings coordinates of the person to center of camera's imaginary axis.
                              if(centre_x<=-20 or centre_x>=20):
                                    if(centre_x<0):
                                          flag=0
                                          Right()
                                          time.sleep(0.025)
                                    elif(centre_x>0):
                                          flag=1
                                          left()
                                          time.sleep(0.025)
                              forward()
                              time.sleep(0.00003125)
                              stop()
                              time.sleep(0.00625)
                        else:
                              stop()
                              time.sleep(0.01)

                  else:
                      
                        time.sleep(0.1)
                        stop()
                        time.sleep(0.1)
      cv2.imshow("draw",frame)
      rawCapture.truncate(0)  # clear the stream in preparation for the next frame
def Greeting ():
		print "Greetings. I am the Monitor of Installation 04. I am 343 Guilty Spark."
		print "gez i havent seen people sense i went offline"
		print "you must either be the reclaimer or a forerunner"
		print "no thats not right ...human...covinant maybe.."
		unknown = input ('who are you!')
		for  (unknown):
	switcher = {
        1: Forerunner
			Forerunner()
        2: Reclaimer,
			Reclaimer()
        3: Human,
			Human()
        4: Covinant,
			Covinant()
        
    }
    print switcher.get(argument, "i dont understand, Who are you")	

def Reclaimer ():
	print "Greetings. I am the Monitor of Installation 04. I am 343 Guilty Spark."
	reclaimerfirstanswer = input('Do you know why you are here?')

	print "Someone has released the Flood. My function is to prevent it from leaving"
	print "this Installation. But I require your assistance. Come. This way."
	forward()
    time.sleep(0.00625)
	stop()
	happy()
	
def happy ():
	print "i am very happy with your help"
def playful ():
	print "your so slow"
def angry ():
	print "you will regret this you herotic"
	forward()
	Right()
	forward()
    time.sleep(0.00625)
def Forerunner():
	print "how long have my masters been hiding in the shadows "

def Human
	print "you are the son of my creator"
def Covinant
	print "do not destroy the rings"






