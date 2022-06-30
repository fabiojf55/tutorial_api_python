import csv
import cv2
import functions_general as fg
import numpy as np

data = open('Data/Sensor_Data/Test_02.csv', newline='')
# print(data)

frames = csv.DictReader(data, delimiter=';')
print(frames)

keypoints_sensors = np.zeros((23,3))

for item in frames:
    img = cv2.imread('Data/black.jpg')

    keypoints_sensors[0][0] = float(item['Pelvis x'].replace(',', '.'))
    keypoints_sensors[0][1] = float(item['Pelvis y'].replace(',', '.'))
    keypoints_sensors[0][2] = float(item['Pelvis z'].replace(',', '.'))
    keypoints_sensors[1][0] = float(item['L5 x'].replace(',', '.'))                         # Not needed
    keypoints_sensors[1][1] = float(item['L5 y'].replace(',', '.'))                         # Not needed
    keypoints_sensors[1][2] = float(item['L5 z'].replace(',', '.'))                         # Not needed
    keypoints_sensors[2][0] = float(item['L3 x'].replace(',', '.'))                         # Not needed
    keypoints_sensors[2][1] = float(item['L3 y'].replace(',', '.'))                         # Not needed
    keypoints_sensors[2][2] = float(item['L3 z'].replace(',', '.'))                         # Not needed
    keypoints_sensors[3][0] = float(item['T12 x'].replace(',', '.'))                        # Not needed
    keypoints_sensors[3][1] = float(item['T12 y'].replace(',', '.'))                        # Not needed
    keypoints_sensors[3][2] = float(item['T12 z'].replace(',', '.'))                        # Not needed
    keypoints_sensors[4][0] = float(item['T8 x'].replace(',', '.'))                         # Not needed
    keypoints_sensors[4][1] = float(item['T8 y'].replace(',', '.'))                         # Not needed
    keypoints_sensors[4][2] = float(item['T8 z'].replace(',', '.'))                         # Not needed
    keypoints_sensors[5][0] = float(item['Neck x'].replace(',', '.'))                       # Not needed
    keypoints_sensors[5][1] = float(item['Neck y'].replace(',', '.'))                       # Not needed
    keypoints_sensors[5][2] = float(item['Neck z'].replace(',', '.'))                       # Not needed
    keypoints_sensors[6][0] = float(item['Head x'].replace(',', '.'))                       # Not needed
    keypoints_sensors[6][1] = float(item['Head y'].replace(',', '.'))                       # Not needed
    keypoints_sensors[6][2] = float(item['Head z'].replace(',', '.'))                       # Not needed
    keypoints_sensors[7][0] = float(item['Right Shoulder x'].replace(',', '.'))
    keypoints_sensors[7][1] = float(item['Right Shoulder y'].replace(',', '.'))
    keypoints_sensors[7][2] = float(item['Right Shoulder z'].replace(',', '.'))
    keypoints_sensors[8][0] = float(item['Right Upper Arm x'].replace(',', '.'))
    keypoints_sensors[8][1] = float(item['Right Upper Arm y'].replace(',', '.'))
    keypoints_sensors[8][2] = float(item['Right Upper Arm z'].replace(',', '.'))
    keypoints_sensors[9][0] = float(item['Right Forearm x'].replace(',', '.'))
    keypoints_sensors[9][1] = float(item['Right Forearm y'].replace(',', '.'))
    keypoints_sensors[9][2] = float(item['Right Forearm z'].replace(',', '.'))
    keypoints_sensors[10][0] = float(item['Right Hand x'].replace(',', '.'))
    keypoints_sensors[10][1] = float(item['Right Hand y'].replace(',', '.'))
    keypoints_sensors[10][2] = float(item['Right Hand z'].replace(',', '.'))
    keypoints_sensors[11][0] = float(item['Left Shoulder x'].replace(',', '.'))
    keypoints_sensors[11][1] = float(item['Left Shoulder y'].replace(',', '.'))
    keypoints_sensors[11][2] = float(item['Left Shoulder z'].replace(',', '.'))
    keypoints_sensors[12][0] = float(item['Left Upper Arm x'].replace(',', '.'))
    keypoints_sensors[12][1] = float(item['Left Upper Arm y'].replace(',', '.'))
    keypoints_sensors[12][2] = float(item['Left Upper Arm z'].replace(',', '.'))
    keypoints_sensors[13][0] = float(item['Left Forearm x'].replace(',', '.'))
    keypoints_sensors[13][1] = float(item['Left Forearm y'].replace(',', '.'))
    keypoints_sensors[13][2] = float(item['Left Forearm z'].replace(',', '.'))
    keypoints_sensors[14][0] = float(item['Left Hand x'].replace(',', '.'))
    keypoints_sensors[14][1] = float(item['Left Hand y'].replace(',', '.'))
    keypoints_sensors[14][2] = float(item['Left Hand z'].replace(',', '.'))
    keypoints_sensors[15][0] = float(item['Right Upper Leg x'].replace(',', '.'))
    keypoints_sensors[15][1] = float(item['Right Upper Leg y'].replace(',', '.'))
    keypoints_sensors[15][2] = float(item['Right Upper Leg z'].replace(',', '.'))
    keypoints_sensors[16][0] = float(item['Right Lower Leg x'].replace(',', '.'))
    keypoints_sensors[16][1] = float(item['Right Lower Leg y'].replace(',', '.'))
    keypoints_sensors[16][2] = float(item['Right Lower Leg z'].replace(',', '.'))
    keypoints_sensors[17][0] = float(item['Right Foot x'].replace(',', '.'))
    keypoints_sensors[17][1] = float(item['Right Foot y'].replace(',', '.'))
    keypoints_sensors[17][2] = float(item['Right Foot z'].replace(',', '.'))
    keypoints_sensors[18][0] = float(item['Right Toe x'].replace(',', '.'))
    keypoints_sensors[18][1] = float(item['Right Toe y'].replace(',', '.'))
    keypoints_sensors[18][2] = float(item['Right Toe z'].replace(',', '.'))
    keypoints_sensors[19][0] = float(item['Left Upper Leg x'].replace(',', '.'))
    keypoints_sensors[19][1] = float(item['Left Upper Leg y'].replace(',', '.'))
    keypoints_sensors[19][2] = float(item['Left Upper Leg z'].replace(',', '.'))
    keypoints_sensors[20][0] = float(item['Left Lower Leg x'].replace(',', '.'))
    keypoints_sensors[20][1] = float(item['Left Lower Leg y'].replace(',', '.'))
    keypoints_sensors[20][2] = float(item['Left Lower Leg z'].replace(',', '.'))
    keypoints_sensors[21][0] = float(item['Left Foot x'].replace(',', '.'))
    keypoints_sensors[21][1] = float(item['Left Foot y'].replace(',', '.'))
    keypoints_sensors[21][2] = float(item['Left Foot z'].replace(',', '.'))
    keypoints_sensors[22][0] = float(item['Left Toe x'].replace(',', '.'))
    keypoints_sensors[22][1] = float(item['Left Toe y'].replace(',', '.'))
    keypoints_sensors[22][2] = float(item['Left Toe z'].replace(',', '.'))

    # Offsets and scaling
    x = 500
    x1 = 300
    y = 500
    cv2.circle(img, (x1, y), 7, (255, 0, 0), cv2.FILLED) # Origin
    # cv2.putText(img, 'Origin', (20, 90), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3, bottomLeftOrigin=False)

    # for i in range(len(keypoints_sensors)):
    #         cv2.circle(img, (int(keypoints_sensors[i][1] * x + x1), int(keypoints_sensors[i][2] * y)), 4,(0, 0, 255), cv2.FILLED)

    for i in range(16):
            cv2.circle(img, (int(keypoints_sensors[i][1] * x + x1), int(keypoints_sensors[i][2] * y)), 4,(0, 0, 255), cv2.FILLED)

    for i in range(7):
            cv2.circle(img, (int(keypoints_sensors[16+i][1] * x + x1), int(keypoints_sensors[16+i][2] * y)), 4,(0, 255, 0), cv2.FILLED)

    # pelvis_x = float(item['Pelvis x'].replace(',', '.'))
    # pelvis_y = float(item['Pelvis y'].replace(',', '.'))
    # pelvis_z = float(item['Pelvis z'].replace(',', '.'))
    # L5_x = float(item['L5 x'].replace(',', '.'))
    # L5_y = float(item['L5 y'].replace(',', '.'))
    # L5_z = float(item['L5 z'].replace(',', '.'))
    # L3_x = float(item['L3 x'].replace(',', '.'))
    # L3_y = float(item['L3 y'].replace(',', '.'))
    # L3_z = float(item['L3 z'].replace(',', '.'))
    # T12_x = float(item['T12 x'].replace(',', '.'))
    # T12_y = float(item['T12 y'].replace(',', '.'))
    # T12_z = float(item['T12 z'].replace(',', '.'))
    # T8_x = float(item['T8 x'].replace(',', '.'))
    # T8_y = float(item['T8 y'].replace(',', '.'))
    # T8_z = float(item['T8 z'].replace(',', '.'))
    # Neck_x = float(item['Neck x'].replace(',', '.'))
    # Neck_y = float(item['Neck y'].replace(',', '.'))
    # Neck_z = float(item['Neck z'].replace(',', '.'))
    # Head_x = float(item['Head x'].replace(',', '.'))
    # Head_y = float(item['Head y'].replace(',', '.'))
    # Head_z = float(item['Head z'].replace(',', '.'))
    # Right_Shoulder_x = float(item['Right Shoulder x'].replace(',', '.'))
    # Right_Shoulder_y = float(item['Right Shoulder y'].replace(',', '.'))
    # Right_Shoulder_z = float(item['Right Shoulder z'].replace(',', '.'))
    # Right_Upper_Arm_x = float(item['Right Upper Arm x'].replace(',', '.'))
    # Right_Upper_Arm_y = float(item['Right Upper Arm y'].replace(',', '.'))
    # Right_Upper_Arm_z = float(item['Right Upper Arm z'].replace(',', '.'))
    # Right_Forearm_x = float(item['Right Forearm x'].replace(',', '.'))
    # Right_Forearm_y = float(item['Right Forearm y'].replace(',', '.'))
    # Right_Forearm_z = float(item['Right Forearm z'].replace(',', '.'))
    # Right_Hand_x = float(item['Right Hand x'].replace(',', '.'))
    # Right_Hand_y = float(item['Right Hand y'].replace(',', '.'))
    # Right_Hand_z = float(item['Right Hand z'].replace(',', '.'))
    # Left_Shoulder_x = float(item['Left Shoulder x'].replace(',', '.'))
    # Left_Shoulder_y = float(item['Left Shoulder y'].replace(',', '.'))
    # Left_Shoulder_z = float(item['Left Shoulder z'].replace(',', '.'))
    # Left_Upper_Arm_x = float(item['Left Upper Arm x'].replace(',', '.'))
    # Left_Upper_Arm_y = float(item['Left Upper Arm y'].replace(',', '.'))
    # Left_Upper_Arm_z = float(item['Left Upper Arm z'].replace(',', '.'))
    # Left_Forearm_x = float(item['Left Forearm x'].replace(',', '.'))
    # Left_Forearm_y = float(item['Left Forearm y'].replace(',', '.'))
    # Left_Forearm_z = float(item['Left Forearm z'].replace(',', '.'))
    # Left_Hand_x = float(item['Left Hand x'].replace(',', '.'))
    # Left_Hand_y = float(item['Left Hand y'].replace(',', '.'))
    # Left_Hand_z = float(item['Left Hand z'].replace(',', '.'))
    # Right_Upper_Leg_x = float(item['Right Upper Leg x'].replace(',', '.'))
    # Right_Upper_Leg_y = float(item['Right Upper Leg y'].replace(',', '.'))
    # Right_Upper_Leg_z = float(item['Right Upper Leg z'].replace(',', '.'))
    # Right_Lower_Leg_x = float(item['Right Lower Leg x'].replace(',', '.'))
    # Right_Lower_Leg_y = float(item['Right Lower Leg y'].replace(',', '.'))
    # Right_Lower_Leg_z = float(item['Right Lower Leg z'].replace(',', '.'))
    # Right_Foot_x = float(item['Right Foot x'].replace(',', '.'))
    # Right_Foot_y = float(item['Right Foot y'].replace(',', '.'))
    # Right_Foot_z = float(item['Right Foot z'].replace(',', '.'))
    # Right_Toe_x = float(item['Right Toe x'].replace(',', '.'))
    # Right_Toe_y = float(item['Right Toe y'].replace(',', '.'))
    # Right_Toe_z = float(item['Right Toe z'].replace(',', '.'))
    # Left_Upper_Leg_x = float(item['Left Upper Leg x'].replace(',', '.'))
    # Left_Upper_Leg_y = float(item['Left Upper Leg y'].replace(',', '.'))
    # Left_Upper_Leg_z = float(item['Left Upper Leg z'].replace(',', '.'))
    # Left_Lower_Leg_x = float(item['Left Lower Leg x'].replace(',', '.'))
    # Left_Lower_Leg_y = float(item['Left Lower Leg y'].replace(',', '.'))
    # Left_Lower_Leg_z = float(item['Left Lower Leg z'].replace(',', '.'))
    # Left_Foot_x = float(item['Left Foot x'].replace(',', '.'))
    # Left_Foot_y = float(item['Left Foot y'].replace(',', '.'))
    # Left_Foot_z = float(item['Left Foot z'].replace(',', '.'))
    # Left_Toe_x = float(item['Left Toe x'].replace(',', '.'))
    # Left_Toe_y = float(item['Left Toe y'].replace(',', '.'))
    # Left_Toe_z = float(item['Left Toe z'].replace(',', '.'))

    # x = 500
    # x1 = 300
    # y = 500
    # # print(float(pelvis_x.replace(',','.'))*x)
    # cv2.circle(img, (int(pelvis_x * x + x1), int(pelvis_z * y)), 4,
    #            (0, 0, 255), cv2.FILLED)
    # cv2.circle(img, (int(float(L5_x.replace(',', '.')) * x + x1), int(float(L5_z.replace(',', '.')) * y)), 4,
    #            (0, 0, 255), cv2.FILLED)
    # cv2.circle(img, (int(float(L3_x.replace(',', '.')) * x + x1), int(float(L3_z.replace(',', '.')) * y)), 4,
    #            (0, 0, 255), cv2.FILLED)
    # cv2.circle(img, (int(float(T12_x.replace(',', '.')) * x + x1), int(float(T12_z.replace(',', '.')) * y)), 4,
    #            (0, 0, 255), cv2.FILLED)
    # cv2.circle(img, (int(float(T8_x.replace(',', '.')) * x + x1), int(float(T8_z.replace(',', '.')) * y)), 4,
    #            (0, 0, 255), cv2.FILLED)
    # cv2.circle(img, (int(float(Neck_x.replace(',', '.')) * x + x1), int(float(Neck_z.replace(',', '.')) * y)), 4,
    #            (0, 0, 255), cv2.FILLED)
    # cv2.circle(img, (int(float(Head_x.replace(',', '.')) * x + x1), int(float(Head_z.replace(',', '.')) * y)), 4,
    #            (0, 0, 255), cv2.FILLED)
    # cv2.circle(img, (int(float(Right_Shoulder_x.replace(',', '.')) * x + x1), int(float(Right_Shoulder_z.replace(',', '.')) * y)), 4,
    #            (255, 0, 0), cv2.FILLED)
    # cv2.circle(img, (int(float(Right_Upper_Arm_x.replace(',', '.')) * x + x1), int(float(Right_Upper_Arm_z.replace(',', '.')) * y)), 4,
    #            (255, 0, 0), cv2.FILLED)
    # cv2.circle(img, (int(float(Right_Forearm_x.replace(',', '.')) * x + x1), int(float(Right_Forearm_z.replace(',', '.')) * y)), 4,
    #            (255, 0, 0), cv2.FILLED)
    # cv2.circle(img, (int(float(Right_Hand_x.replace(',', '.')) * x + x1), int(float(Right_Hand_z.replace(',', '.')) * y)), 4,
    #            (255, 0, 0), cv2.FILLED)
    # cv2.circle(img, (int(float(Left_Shoulder_x.replace(',', '.')) * x + x1), int(float(Left_Shoulder_z.replace(',', '.')) * y)), 4,
    #            (0, 255, 0), cv2.FILLED)
    # cv2.circle(img, (int(float(Left_Upper_Arm_x.replace(',', '.')) * x + x1), int(float(Left_Upper_Arm_z.replace(',', '.')) * y)), 4,
    #            (0, 255, 0), cv2.FILLED)
    # cv2.circle(img, (int(float(Left_Forearm_x.replace(',', '.')) * x + x1), int(float(Left_Forearm_z.replace(',', '.')) * y)), 4,
    #            (0, 255, 0), cv2.FILLED)
    # cv2.circle(img, (int(float(Left_Hand_x.replace(',', '.')) * x + x1), int(float(Left_Hand_z.replace(',', '.')) * y)), 4,
    #            (0, 255, 0), cv2.FILLED)
    # cv2.circle(img, (int(float(Right_Upper_Leg_x.replace(',', '.')) * x + x1), int(float(Right_Upper_Leg_z.replace(',', '.')) * y)), 4,
    #            (255, 165, 0), cv2.FILLED)
    # cv2.circle(img, (int(float(Right_Lower_Leg_x.replace(',', '.')) * x + x1), int(float(Right_Lower_Leg_z.replace(',', '.')) * y)), 4,
    #            (255, 165, 0), cv2.FILLED)
    # cv2.circle(img, (int(float(Right_Foot_x.replace(',', '.')) * x + x1), int(float(Right_Foot_z.replace(',', '.')) * y)), 4,
    #            (255, 165, 0), cv2.FILLED)
    # cv2.circle(img, (int(float(Right_Toe_x.replace(',', '.')) * x + x1), int(float(Right_Toe_z.replace(',', '.')) * y)), 4,
    #            (255, 165, 0), cv2.FILLED)
    # cv2.circle(img, (int(float(Left_Upper_Leg_x.replace(',', '.')) * x + x1), int(float(Left_Upper_Leg_z.replace(',', '.')) * y)), 4,
    #            (255, 0, 255), cv2.FILLED)
    # cv2.circle(img, (int(float(Left_Lower_Leg_x.replace(',', '.')) * x + x1), int(float(Left_Lower_Leg_z.replace(',', '.')) * y)), 4,
    #            (255, 0, 255), cv2.FILLED)
    # cv2.circle(img, (int(float(Left_Foot_x.replace(',', '.')) * x + x1), int(float(Left_Foot_z.replace(',', '.')) * y)), 4,
    #            (255, 0, 255), cv2.FILLED)
    # cv2.circle(img, (int(float(Left_Toe_x.replace(',', '.')) * x + x1), int(float(Left_Toe_z.replace(',', '.')) * y)), 4,
    #            (255, 0, 255), cv2.FILLED)

    flippedImg = cv2.flip(img, 0)

    new_img = fg.scale_img(flippedImg)
    cv2.imshow('asdf', new_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

