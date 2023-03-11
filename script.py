from keras.models import load_model
import cv2
import numpy as np
import mediapipe as mp



letter = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N',
                        'O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']

# you need to download the model with this link :
# https://drive.google.com/drive/folders/1jZwygJjX02akEte82-3JEjjlpmglEJGR?usp=sharing
model = load_model('my_model_60_new.h5')

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    image_height, image_width, _ = img.shape

    if results.multi_hand_landmarks:
        
        for hand_landmarks in results.multi_hand_landmarks:
            sign_img = img/255

            x = [landmark.x for landmark in hand_landmarks.landmark]
            y = [landmark.y for landmark in hand_landmarks.landmark]

            
            center = np.array([np.mean(x) * image_width, np.mean(y) * image_height]).astype('int64')

            try :
            
                cropped_image = cv2.rectangle(img, (center[0] - 240, center[1] - 240),(center[0] + 240, center[1] + 240), (0, 0, 255), 2)
                cropped_img = img[center[1]-240:center[1]+240, center[0]-240:center[0]+240]
                full_img = cropped_image

                sign_img = sign_img[center[1]-240:center[1]+240, center[0]-240:center[0]+240]

                sign_img = cv2.resize(sign_img, (64,64))
                sign_img = sign_img.reshape((1,)+sign_img.shape)
                pre = model.predict(sign_img,verbose=False)
                
                print(pre)
                print([np.argmax(pre)], np.max(pre))



                cv2.putText(img, letter[np.argmax(pre)], (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
                cv2.putText(img, str(round(np.max(pre), 2)), (10,140), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)

            except:
                pass

    cv2.imshow("Image", img)
    cv2.waitKey(1)

