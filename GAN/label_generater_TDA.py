import os, sys
import csv
import torchvision.transforms as transforms
# 打开文件
path = "/Users/lichen/Desktop/test/models_data/" # please put the data resources path here
dirs = os.listdir( path )

storagelist = []
image_size = 64
model_names = []
# 输出所有文件和文件夹
for file in dirs:
    if file.startswith('.'):
        pass
    else:
        model_names.append(file)


global_index = 0


label_list =['UAV Model_name',
             'Number of propellers',
             'Color of main body',
             #######################
             'Independent lens module',
             'Independent floor stand or not',
             'Shape of floor stand',
             'With or without propeller protection',
             'The shape of the main body shell',
             #######################
             "Wing Bracket Topology",
             'Single color design - propellers/body/bottom',
             'Wing brackets shape',
             'Propeller shape',
             'Wing position indicator',
             ########################
             'Heat sink position',
             'Printing logo position',
             "Propeller color",
             "Decorations in propellers",
             "Decorations in wing bracket",
             'Heat sink shape',
             ]



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
            storagelist.append([root_path +'/'+ img_path,model_name])

###  DJI_Mavic_Mini  ####

        elif model_name == 'DJI_Mavic_Mini':
            storagelist.append([root_path +'/'+ img_path,model_name])

###  DJI_Phantom_4_pro_3D_model  ####

        elif model_name == 'DJI_Phantom_4_pro_3D_model':
            storagelist.append([root_path +'/'+ img_path,model_name])

###  DJI_Phantom_3_Pro  ####

        elif model_name == 'DJI_Phantom_3_Pro':
            storagelist.append([root_path +'/'+ img_path,model_name])

###  DJI_Phantom_4  ####

        elif model_name == 'DJI_Phantom_4':
            storagelist.append([root_path +'/'+ img_path,model_name])

###  3DR_Solo_Drone_for_Element_3D  ####

        elif model_name == '3DR_Solo_Drone_for_Element_3D':
            storagelist.append([root_path +'/'+ img_path,model_name])

###  Carbon_Fiber_Drone_-_Unmanned_Aerial_Vehicle  ####

        elif model_name == 'Carbon_Fiber_Drone_-_Unmanned_Aerial_Vehicle':
            storagelist.append([root_path +'/'+ img_path,model_name])

###  DJI_Inspire_1  ####

        elif model_name == 'DJI_Inspire_1':
            storagelist.append([root_path +'/'+ img_path,model_name])

###  DJI_Inspire_2_for_Element_3D  ####

        elif model_name == 'DJI_Inspire_2_for_Element_3D':
            storagelist.append([root_path +'/'+ img_path,model_name])

###  DJI_Mavic_Pro_for_Element_3D  ####

        elif model_name == 'DJI_Mavic_Pro_for_Element_3D':
            storagelist.append([root_path +'/'+ img_path,model_name])

###  Generic_Branding_Technology_-_Drone  ####

        elif model_name == 'Generic_Branding_Technology_-_Drone':
            storagelist.append([root_path +'/'+ img_path,model_name])

###  DJI_Spark_Drone_for_Element_3D  ####

        elif model_name == 'DJI_Spark_Drone_for_Element_3D':
            storagelist.append([root_path +'/'+ img_path,model_name])

###  DJI_Spark_drone_Low_poly_Animated  ####

        elif model_name == 'DJI_Spark_drone_Low_poly_Animated':
            storagelist.append([root_path +'/'+ img_path,model_name])

###  Customized_Drone  ####

        elif model_name == 'Customized_Drone':
            storagelist.append([root_path +'/'+ img_path,model_name])



###  unknown uav  ####
        else:storagelist.append([root_path +'/'+ img_path,model_name,str(4)+';'+str(global_index)])
        global_index += 1


with open('/Users/lichen/Desktop/test/data_labels.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerows(storagelist)

