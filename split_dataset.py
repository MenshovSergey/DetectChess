
import numpy as np
import cv2
import os
x_train = np.load("/home/sergej/idchess/xtrain.npy")
y_train = np.load("/home/sergej/idchess/ytrain.npy")

y_train = [x * 256 for x in y_train]
out_path = '/home/sergej/idchess/images'
for i, v in enumerate(x_train[0:100]):
    # img_train = cv2.circle(v,(int(y_train[i][4]), int(y_train[i][5])),20, color=128 )
    left = min(y_train[i][1], y_train[i][3], y_train[i][5], y_train[i][7])
    right = max(y_train[i][1], y_train[i][3], y_train[i][5], y_train[i][7])

    bottom = min(y_train[i][0], y_train[i][2], y_train[i][4], y_train[i][6])
    top = max(y_train[i][0], y_train[i][2], y_train[i][4], y_train[i][6])
    img_train = v[int(left):int(right),int(bottom):int(top) ]
    # img_train = v[int(y_train[i][1]):int(y_train[i][5]),int(y_train[i][0]):int(y_train[i][4]) ]
    cv2.imwrite(os.path.join(out_path, str(i)+".png"), img_train)

for i, v in enumerate(x_train[100:200]):
    cv2.imwrite(os.path.join('/home/sergej/idchess/test', str(i) + ".png"), v)