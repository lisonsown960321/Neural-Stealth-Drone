import torch
import random
import csv
storagelist = []

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

storagelist.append(label_list)

model_name = 'Autel_X-Star_Premium_Drone_for_Element_3D'
storagelist.append([model_name,str(4),'red',
                           'Under-hung','yes','Square bracket','none','Spherical shape',
                           'X shape','2-color','One-piece Cylindrical','Double blade','black',
                           'Bottom and wing bracket','None','black','none','none','long strip'
                        ])

###  DJI_Mavic_Mini  ####

model_name = 'DJI_Mavic_Mini'
storagelist.append([model_name,str(4),'grey',
                          'Front-hung','none','Wing embedded','none','Cuboid shape',
                          'Insect shape','2-color','Convertible Cuboid','Double blade','none',
                          'Bottom and sides','body and wing bracket','black','grey','none','Honeycomb, long strip',
                        ])

###  DJI_Phantom_4_pro_3D_model  ####

model_name = 'DJI_Phantom_4_pro_3D_model'
storagelist.append([model_name,str(4),'white',
                          'Under-hung','yes','Square bracket','none','Cube shape',
                          'X shape','1-color','One-piece Cylindrical','Double blade','red-green',
                          'Bottom and wing bracket','body and bottom','white','none','none','Honeycomb',
                        ])

###  DJI_Phantom_3_Pro  ####

model_name = 'DJI_Phantom_3_Pro'
storagelist.append([model_name,str(4),'white',
                          'Under-hung','yes','Square bracket','none','hidden Cuboid shape',
                          'X shape','1-color','One-piece Cylindrical','Double blade','none',
                          'Bottom and wing bracket','body and front','white','none','2 golden lines in front','Long strip',
                        ])

###  DJI_Phantom_4  ####

model_name = 'DJI_Phantom_4'
storagelist.append([model_name,str(4),'white',
                          'Under-hung','yes','Square bracket','none','hidden shape',
                          'X shape','2-color','One-piece Cylindrical','Double blade','red',
                          'Bottom and wing bracket','body and bottom','white','none','none','Honeycomb, long strip',
])

###  3DR_Solo_Drone_for_Element_3D  ####

model_name = '3DR_Solo_Drone_for_Element_3D'
storagelist.append([model_name,str(4),'black',
                          'Under-hung','yes','4-feet straight','none','Cuboid shape',
                          'I shape','1-color','One-piece Polyline','Double blade','red and black',
                          'None','body and bottom','black','none','none','none',
])

###  Carbon_Fiber_Drone_-_Unmanned_Aerial_Vehicle  ####

model_name = 'Carbon_Fiber_Drone_-_Unmanned_Aerial_Vehicle'
storagelist.append([model_name,str(4),'black',
                          'Front-hung','none','Wing embedded','none','Cuboid shape',
                          'Insect shape','1-color','One-piece Trapezoid','Double blade','blue',
                          'Sides','body and two sides','black','2 blue lines','none','Square Shape',
])

###  DJI_Inspire_1  ####

model_name = 'DJI_Inspire_1'
storagelist.append([model_name,str(4),'white',
                          'Front-hung','yes','Wing embedded triangle','none','Vertical rhombus',
                          'H shape','2-color','Splicing T shape','Double blade','none',
                          'Body and Hollow-carved design in sides','none','black','none','carbon','Y Shape',
])

###  DJI_Inspire_2_for_Element_3D  ####

model_name = 'DJI_Inspire_2_for_Element_3D'
storagelist.append([model_name,str(4),'grey',
                          'Front-hung','yes','Wing embedded Y shape','none','Vertical rhombus',
                          'V-H shape','2-color','Splicing T shape','Double blade','none',
                          'body and Hollow-carved design in sides','none','black','none','none','Y Shape',
])

###  DJI_Mavic_Pro_for_Element_3D  ####

model_name = 'DJI_Mavic_Pro_for_Element_3D'
storagelist.append([model_name,str(4),'Navy',
                          'Front-hung','none','Wing embedded','none','Cuboid shape',
                          'Insect shape','2-color','Convertible Cuboid','Double blade','black',
                          'Tail and sides','Tail and front bracket','black','grey','one golden line in front','Square shape',
])

###  Generic_Branding_Technology_-_Drone  ####

model_name = 'Generic_Branding_Technology_-_Drone'
storagelist.append([model_name,str(4),'white',
                          'Under-hung','yes','Square bracket','none','hidden shape',
                          'X shape','1-color','One-piece Cylindrical','Double blade','none',
                          'none','none','white','none','none','none',
])

###  DJI_Spark_Drone_for_Element_3D  ####

model_name = 'DJI_Spark_Drone_for_Element_3D'
storagelist.append([model_name,str(4),'red',
                          'Front-hung','none','Wing embedded','none','Cuboid shape',
                          'Insect shape','2-color','One-piece Trapezoid','Double blade','black',
                          'Sides','body','black','grey','none','Square shape',
])

###  DJI_Spark_drone_Low_poly_Animated  ####

model_name = 'DJI_Spark_drone_Low_poly_Animated'
storagelist.append([model_name,str(4),'grey',
                          'hidden/none','none','Wing embedded','Sector shape','Cuboid shape',
                          'Insect shape','1-color','One-piece Trapezoid','Double blade','red and green',
                          'Sides','none','grey','none','none','Square shape',
])

###  Customized_Drone  ####

model_name = 'Customized_Drone'
storagelist.append([model_name,str(4),'Metallic',
                          'none','yes','T shape','none','Round shape',
                          'Cruciform','2-color','One-piece straight','Double blade','none',
                          'none','none','black','none','black lines','none',
])

with open('/Users/lichen/Desktop/test/model_labels.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerows(storagelist)