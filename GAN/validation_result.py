import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

root_gt_b_loss = '/Users/lichen/Desktop/test/validation_performance1/model_GT_b_loss.csv'
root_r_b_loss = '/Users/lichen/Desktop/test/validation_performance1/model_R_b_loss.csv'
root_gt_test = '/Users/lichen/Desktop/test/validation_performance1/model_loss_GT_test.csv'
root_gt_train = '/Users/lichen/Desktop/test/validation_performance1/model_loss_GT_train.csv'
root_r_test = '/Users/lichen/Desktop/test/validation_performance1/model_loss_R_test.csv'
root_r_train = '/Users/lichen/Desktop/test/validation_performance1/model_loss_R_train.csv'

gt_b_loss = []
r_b_loss = []
gt_test = []
gt_train = []
r_test = []
r_train = []


with open(root_gt_b_loss,'r') as f:
    csv_reader = csv.reader(f)
    for i in csv_reader:
        gt_b_loss.append(i)
with open(root_r_b_loss,'r') as f:
    csv_reader = csv.reader(f)
    for i in csv_reader:
        r_b_loss.append(i)
with open(root_r_train,'r') as f:
    csv_reader = csv.reader(f)
    for i in csv_reader:
        r_train = i
with open(root_r_test,'r') as f:
    csv_reader = csv.reader(f)
    for i in csv_reader:
        r_test = i
with open(root_gt_train,'r') as f:
    csv_reader = csv.reader(f)
    for i in csv_reader:
        gt_train = i
with open(root_gt_test,'r') as f:
    csv_reader = csv.reader(f)
    for i in csv_reader:
        gt_test = i


def convert_number(a):
    ret = []
    for i in a:
        ret.append(float(i))
    return ret


def convert_number_2d(a, index = 0):
    ret = []
    if index == 0:
        for i in a:
            ret.append(1 - (float(i[0])+float(i[1])))
        return ret
    elif index == 1:
        for i in a:
            ret.append(float(i[0]))
        return ret
    elif index == 2:
        for i in a:
            ret.append(float(i[1]))
        return ret
    else:return('error')

def logc(a):
    a = np.asarray(a)
    a = np.log10(a)
    return a
gt_train = convert_number(gt_train)
r_train = convert_number(r_train)
gt_test = convert_number(gt_test)
r_test = convert_number(r_test)
fig = plt.figure(figsize=(30, 40))
columns = 3
row = 1


ax1 = plt.subplot(311)
x = np.arange(len(gt_train))
ax1.plot(x,gt_train,color="r",label='GAN-TDA',linewidth=8)
ax1.plot(x,r_train,color="b",label='Randomly',linewidth=8)
ax1.legend(fontsize=42)
ax1.set_xlabel('epochs',fontdict={'family' : 'Times New Roman', 'size'   : 42})
ax1.set_ylabel('loss',fontdict={'family' : 'Times New Roman', 'size'   : 42})
ax1.tick_params(labelsize=42)
ax1.set_title('A) Training Loss (BCE)',fontsize=42)
ax1.set_yscale('log')
# ax1.yscale(u'log')


ax2 = plt.subplot(312)
x = np.arange(len(gt_test))
ax2.plot(x,gt_test,color="r",label='GAN-TDA',linewidth=8)
ax2.plot(x,r_test,color="b",label='Randomly',linewidth=8)
ax2.legend(fontsize=42)
ax2.set_xlabel('epochs',fontdict={'family' : 'Times New Roman', 'size'   : 42})
ax2.set_ylabel('loss',fontdict={'family' : 'Times New Roman', 'size'   : 42})
ax2.tick_params(labelsize=42)
ax2.set_title('B) Validation Loss (BCE)',fontsize=42)
ax2.set_yscale('log')
# ax2.plot(logy=True)

gt_b_loss1 = convert_number_2d(gt_b_loss,0)
r_b_loss1 = convert_number_2d(r_b_loss,0)
ax3 = plt.subplot(313)
x = np.arange(len(gt_b_loss))
ax3.plot(x,gt_b_loss1,color="r",label='GAN-TDA',linewidth=8)
ax3.plot(x,r_b_loss1,color="b",label='Randomly',linewidth=8)
ax3.legend(fontsize=42)
ax3.set_xlabel('epochs',fontdict={'family' : 'Times New Roman', 'size'   : 42})
ax3.set_ylabel('Precision',fontdict={'family' : 'Times New Roman', 'size'   : 42})
ax3.tick_params(labelsize=42)
ax3.set_title('C) Validation Discriminative Precision',fontsize=42)
ax3.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.subplots_adjust(wspace =0, hspace =0.3)#调整子图间距
# plt.show()
plt.savefig('/Users/lichen/Desktop/result.png')