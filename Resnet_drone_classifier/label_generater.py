import os, sys
import csv
import torchvision.transforms as transforms
# 打开文件
# path = "C:/Users/s324652/OneDrive - Cranfield University/Desktop" # please put the data resources path here
path = "C:/Users/s324652/OneDrive - Cranfield University/Desktop/veli/"
store_path = "C:/Users/s324652/Desktop/yolov2/DroneDetector/"

dirs = os.listdir(path)

storagelist = []
image_size = 224
model_names = []
# 输出所有文件和文件夹
for file in dirs:
    if file.startswith('.'):
        pass
    else:
        model_names.append(file)


global_index = 0


for model_name in model_names:
    root_path = path + str(model_name)
    dirs = os.listdir(root_path)
    dirs1 = []
    for kk in dirs:
        if kk.endswith('.png'):
            dirs1.append(kk)
        elif kk.endswith('.DS_Store'):
            print(kk)
        else:pass

    for img_path in dirs1:
###  Autel_X-Star_Premium_Drone_for_Element_3D  ####

        if model_name == 'Autel_X-Star_Premium_Drone_for_Element_3D':
            storagelist.append([root_path +'/'+ img_path,0])

###  DJI_Mavic_Mini  ####

        elif model_name == 'DJI_Mavic_Mini':
            storagelist.append([root_path +'/'+ img_path,1])

###  DJI_Phantom_4_pro_3D_model  ####

        elif model_name == 'DJI_Phantom_4_pro_3D_model':
            storagelist.append([root_path +'/'+ img_path,2])

###  DJI_Phantom_3_Pro  ####

        elif model_name == 'DJI_Phantom_3_Pro':
            storagelist.append([root_path +'/'+ img_path,3])

###  DJI_Phantom_4  ####

        elif model_name == 'DJI_Phantom_4':
            storagelist.append([root_path +'/'+ img_path,4])

###  3DR_Solo_Drone_for_Element_3D  ####

        elif model_name == '3DR_Solo_Drone_for_Element_3D':
            storagelist.append([root_path +'/'+ img_path,5])

###  Carbon_Fiber_Drone_-_Unmanned_Aerial_Vehicle  ####

        elif model_name == 'Carbon_Fiber_Drone_-_Unmanned_Aerial_Vehicle':
            storagelist.append([root_path +'/'+ img_path,6])

###  DJI_Inspire_1  ####

        elif model_name == 'DJI_Inspire_1':
            storagelist.append([root_path +'/'+ img_path,7])

###  DJI_Inspire_2_for_Element_3D  ####

        elif model_name == 'DJI_Inspire_2_for_Element_3D':
            storagelist.append([root_path +'/'+ img_path,8])

###  DJI_Mavic_Pro_for_Element_3D  ####

        elif model_name == 'DJI_Mavic_Pro_for_Element_3D':
            storagelist.append([root_path +'/'+ img_path,9])

###  Generic_Branding_Technology_-_Drone  ####

        elif model_name == 'Generic_Branding_Technology_-_Drone':
            storagelist.append([root_path +'/'+ img_path,10])

###  DJI_Spark_Drone_for_Element_3D  ####

        elif model_name == 'DJI_Spark_Drone_for_Element_3D':
            storagelist.append([root_path +'/'+ img_path,11])

###  DJI_Spark_drone_Low_poly_Animated  ####

        elif model_name == 'DJI_Spark_drone_Low_poly_Animated':
            storagelist.append([root_path +'/'+ img_path,12])

###  Customized_Drone  ####

        elif model_name == 'Customized_Drone':
            storagelist.append([root_path +'/'+ img_path,13])



# ###  unknown uav  ####
#         else:
#             xx_name = 'designed_drone'
#             storagelist.append([root_path +'/'+ img_path,14])
#         global_index += 1


with open('path', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerows(storagelist)

