import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_blue = np.array([100, 0, 0])
upper_blue = np.array([255, 100, 100])

lower_blue_1 = np.array([50, 100, 90])
upper_blue_1 = np.array([140, 255, 255])

lower_blue_2 = np.array([60, 120, 100])
upper_blue_2 = np.array([80, 255, 255])

while True:
  
    ret, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    

    mask_1 = cv2.inRange(hsv, lower_blue_1, upper_blue_1)
    mask_2 = cv2.inRange(hsv, lower_blue_2, upper_blue_2)
    mask_3 = cv2.inRange(hsv,lower_blue,upper_blue)
    

    mask = cv2.bitwise_or(mask_1, mask_2,mask_3)
    
    result = cv2.bitwise_and(frame, frame, mask=mask)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2  

    for contour in contours:
        if cv2.contourArea(contour) > 500:  
            (x, y, w, h) = cv2.boundingRect(contour)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, "MAVI", (x, y -10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA )

               

    
    
    cv2.imshow('Orijinal Görüntü', frame)
    cv2.imshow('Mavi Obje Tespiti', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


   
cap.release()
cv2.destroyAllWindows()