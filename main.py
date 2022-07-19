

import os
import cv2
import numpy as np

# Load datasets
path_test = "data\Original Images\Testing Set"

dataCitra = []

def preprocessing(img):
    hasil = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    hasil2 = cv2.dilate(hasil, kernel, iterations=2)
    return hasil2


def searchFovea(img):

    try:
        lingkaran = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50, param1=60, param2=10, minRadius=20, maxRadius=30)
        lingkaran = np.uint16(np.around(lingkaran))
    except:
        return img


    for i in lingkaran[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 1, (0, 0, 255), 3)
        print(i[0],' dan ',i[1])
        # print("area makula ", i[1], ":", ((4 * i[2]) + i[1]), "  ", (i[0] - (6 * i[2])), ":", (i[0] - (2 * i[2])))

        # ROI besar
        if i[0] < 214 :
            start_point = ((i[0] + (6 * i[2])),i[1])
            end_point = ((i[0] + (2 * i[2])),((4 * i[2]) + i[1]))
            tebal = 0
            cv2.rectangle(img, start_point, end_point, (0, 255, 0), tebal)
            im_crop = img[(i[1]+1):((4 * i[2]) + i[1]), (i[0] + (2 * i[2])):(i[0] + (6 * i[2])+1)]
        else :
            start_point = ((i[0] - (2 * i[2])), i[1])
            end_point = ((i[0] - (6 * i[2])), ((4 * i[2]) + i[1]))
            tebal = 0
            cv2.rectangle(img, start_point, end_point, (0, 255, 0), tebal)
            im_crop = img[(i[1] + 1):((4 * i[2]) + i[1]), (i[0] - (6 * i[2]) + 1):(i[0] - (2 * i[2]))]

        # ROI kecil
        # if i[0] < 214 :
        #     start_point = ((i[0] + (5 * i[2])),i[1])
        #     end_point = ((i[0] + (3 * i[2])),((2 * i[2]) + i[1]))
        #     tebal = 0
        #     cv2.rectangle(img, start_point, end_point, (0, 255, 0), tebal)
        #     im_crop = img[(i[1]+1):((2 * i[2]) + i[1]), (i[0] + (3 * i[2])):(i[0] + (5 * i[2])+1)]
        # else :
        #     start_point = ((i[0] - (3 * i[2])), i[1])
        #     end_point = ((i[0] - (5 * i[2])), ((2 * i[2]) + i[1]))
        #     tebal = 0
        #     cv2.rectangle(img, start_point, end_point, (0, 255, 0), tebal)
        #     im_crop = img[(i[1] + 1):((2 * i[2]) + i[1]), (i[0] - (5 * i[2]) + 1):(i[0] - (3 * i[2]))]

        equ = cv2.equalizeHist(im_crop)
        hasil = cv2.GaussianBlur(equ,(5,5),0)

        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(hasil)
        cv2.circle(im_crop, minLoc, 1, (255, 0, 0), 2)
        print(minLoc)
        krgA = 10
        krgB = 10
        if minLoc[0] == 0 & minLoc[1] == 0:
            cv2.circle(im_crop, (0, 0), 1, (255, 255, 255), 3)
            return img
        elif minLoc[0] < 10 :
            krgA = minLoc[0]
        elif minLoc[1] < 10 :
            krgB = minLoc[1]
        im_crop2 = im_crop[(minLoc[1]-krgB):(minLoc[1]+10), (minLoc[0]-krgA):(minLoc[0]+10)]

        kernel = np.ones((5, 5), np.uint8)
        try:
            hasil2 = cv2.dilate(im_crop2, kernel, iterations=2)
        except:
            hasil2 = im_crop2

        thresh = cv2.adaptiveThreshold(hasil2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 3)
        # ret, thresh = cv2.threshold(hasil2, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        for c in contours:
            # calculate moments for each contour
            M = cv2.moments(c)

            # calculate x,y coordinate of center
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except:
                cX = 0
                cY = 0
            cv2.circle(im_crop2, (cX, cY), 1, (255, 255, 255), 3)
            print(cX," : ",cY)
            # print("Titik Tengah Makula ",cX+1+(i[0] - (6 * i[2]))," : ",cY+1+i[1])
        break

    return img





def create_data():
    for img in os.listdir(path_test):
        try:
            img_array = cv2.imread(os.path.join(path_test, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (
            np.uint16(np.around(img_array.shape[1] * 10 / 100)), np.uint16(np.around(img_array.shape[0] * 10 / 100))),
                                   interpolation=cv2.INTER_AREA)  # resize to normalize data size
            dataCitra.append(preprocessing(new_array))  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass


create_data()
bantu = 0
cv2.imshow("hasil",searchFovea(dataCitra[0]))
cv2.waitKey(0)
cv2.destroyAllWindows()
# for i in dataCitra:
#     dir = 'M:\Study\PROJECT MAKULA\hasil'
#     os.chdir(dir)
#     filename = 'hasil'+str(bantu)+'.jpg'
#     cv2.imwrite(filename, searchFovea(i))
#     bantu+=1


