import os, sys
import csv
import torchvision.transforms as transforms
# 打开文件
path = "/Users/lichen/Desktop/design_drone_org_verification_nb" # please put the data resources path here
dirs = os.listdir( path )

storagelist = []
image_size = 64
model_names = []

global_index = 0


root_path = path
dirs = os.listdir(root_path)
dirs1 = []
for kk in dirs:
    if kk.endswith('.png'):
        dirs1.append(kk)
    elif kk.endswith('.DS_Store'):
        print(kk)
    else:pass

for img_path in dirs1:
    print(img_path)
    storagelist.append([root_path +'/'+ img_path,'design_drone'])
    global_index += 1


with open('/Users/lichen/Desktop/test/dronedesign.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerows(storagelist)

