from itertools import count
import matplotlib.pyplot as plt
import pandas as pd
import cv2


training_list=[]
ok_list=[]
ng_list=[]

training_file=open('test_train.txt','r')
ok_file=open('test_ok.txt','r')
ng_file=open('test_ng.txt','r')

while True:
    rr=training_file.readline()
    if rr=='':
        break
    training_list.append(float(rr))

while True:
    rr=ok_file.readline()
    if rr=='':
        break
    ok_list.append(float(rr))

while True:
    rr=ng_file.readline()
    if rr=='':
        break
    ng_list.append(float(rr))



loss_threshold=0.8   #調整 loss threshold 的值

value_list=ok_list

#0-2.8
intervel_n=100
intervel=2.8/intervel_n

train_count_list=[]
ok_count_list=[]
ng_count_list=[]
for i in range(intervel_n):
    train_count_list.append(0)
    ok_count_list.append(0)
    ng_count_list.append(0)

for i in range(len(training_list)):
    index = int(training_list[i]/intervel)
    train_count_list[index]+=1

for i in range(len(ok_list)):
    index = int(ok_list[i]/intervel)
    ok_count_list[index]+=1

for i in range(len(ng_list)):
    index = int(ng_list[i]/intervel)
    ng_count_list[index]+=30        #正常來說是加 1 #但因為只加 1 幅度太緩看不出來(因為ng資料只有89個)

plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='both')

loss_value=[]
value=0
for i in range(intervel_n):
    loss_value.append(value)
    value+=intervel

plt.plot(loss_value,train_count_list,color='blue')
plt.plot(loss_value,ok_count_list,color='green')
plt.plot(loss_value,ng_count_list,color='red')

plt.vlines(loss_threshold,0,1750,color="black",linewidth=2)


#plt.title('Title',fontsize=50)

#plt.xticks(fontsize=20)
#plt.yticks(fontsize=30)

plt.xlabel('loss')
plt.ylabel('quantity')

plt.savefig('test.jpg')
plt.close()

print()
print('ok smaple:'+str(len(ok_list)))
print('ng smaple:'+str(len(ng_list)))

ok_true=0
ok_false=0
for i in range(len(ok_list)):
    if ok_list[i]<loss_threshold:
        ok_true+=1
    else:
        ok_false+=1

ng_true=0
ng_false=0
for i in range(len(ng_list)):
    if ng_list[i]<loss_threshold:
        ng_true+=1
    else:
        ng_false+=1
print()
print('ok_true:'+str(ok_true))
print('ok_flase:'+str(ok_false))
print('ng_true:'+str(ng_true))
print('ng_false:'+str(ng_false))