#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:18:54 2019

@author: yancykahn
"""
import os
#自动配置所需环境
first = input("是否需要配置环境? y/n (yes or no)")
if first == 'y':
    os.system("pip3 install -Ur requirement.txt")

from matplotlib import pyplot as plt
import cv2
import pyttsx3
from PIL import Image
import json
import numpy as np
from multiprocessing import Process

pid = Process()

def Say(string):
    audio = pyttsx3.init()
    audio.say(string)
    audio.runAndWait()

def SaySth(string):
    global pid
    if pid.is_alive() :
        pid.terminate()
    pid = Process(target=Say,args=(string,))
    pid.start()

#json 读取写入
def json_write(json_str, JsonPath=r"FaceDataConfig/name.json"):
    with open(JsonPath,"w") as f:
        json.dump(json_str, f)

def json_read(JsonPath=r"FaceDataConfig/name.json"):
    with open(JsonPath,"r") as f:
        json_str = json.load(fp=f)
        return json_str


def register_status_init():
    name2id=json_read()
    status = {}
    for i in range(len(name2id)):
        status[str(name2id[str(i)])] = "False"
    
    if (os.path.exists('FaceDataConfig')): # 判断是否存在目录
        pass
    else:
        os.mkdir('FaceDataConfig')
    json_write(status, JsonPath="FaceDataConfig/status.json")
    
    
facexml = r"haarcascade_frontalface_default.xml"

#获取特征图
def GetFeature():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(facexml) # 加载人脸特征库
    
    if (os.path.exists('FaceDataConfig/')): # 判断是否存在目录
        pass
    else:
        os.mkdir('FaceDataConfig/')
    
    name2id = {}
    
    if (os.path.exists('FaceDataConfig/name.json')): # 判断是否存在目录
        name2id=json_read()
    else:
        open(r"FaceDataConfig/name.json","w")
    
    
    SaySth("请输入你的名字")
    face_id = input('\n enter user name:')
    
    # id 和 name 一一对应
    name2id_flag = True
    for i in range(len(name2id)):
        if name2id[str(i)] == face_id:
            name2id_flag = False
            break
        
    if name2id_flag:
        name2id[str(len(name2id))]= face_id#添加映射
        
    json_write(name2id) #写入json
    SaySth("获取脸部特征, 请保持你的脸在窗口中并稍等片刻")
    if (os.path.exists('FaceData/')): # 判断是否存在目录
        pass
    else:
        os.mkdir('FaceData/')
    
    if (os.path.exists('./FaceData/'+str(face_id))): # 判断是否存在目录
        pass
    else:
        os.mkdir('./FaceData/'+str(face_id)+'/')
    
    count = (len(os.listdir('./FaceData/'+face_id)))
    tcount = 0
    while(True):
        ret, frame = cap.read() # 读取一帧的图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 转灰
    
        faces = face_cascade.detectMultiScale(gray, 
            scaleFactor = 1.15, 
            minNeighbors = 5, 
            minSize = (5, 5)
            ) # 检测人脸

        Area=0.0
        test_img=[]
        xx,yy,ww,hh = 0,0,0,0
        for(x, y, w, h) in faces:
            if (w*h > Area and float(w + h) > 300.0):
                test_img=gray[y:y+h, x:x+w]
                xx,yy,ww,hh = x,y,w,h
                Area = w*h
            
        
        if len(test_img) != 0:
            plt.imshow(test_img)
            plt.show()
            cv2.rectangle(frame, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2) # 用矩形圈出人脸
            cv2.imwrite('./FaceData/'+str(face_id)+'/User.'+str(face_id)+'.'+str(count)+'.jpg', test_img) #保存最大特征图
            count = count + 1
            tcount = tcount + 1
            
        
        cv2.imshow('video', frame)
        print(tcount)
        if tcount == 100:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release() # 释放摄像头
    cv2.destroyAllWindows()


#获取特征和id
def GetFeatureAndLabels(path):
    Features = [f for f in os.listdir(path)]
    imgPaths = []
    FaceSamples = []
    ids = []
    names = []
    
    name2id = json_read()
    for feature in Features:
        test_path = os.path.join(path, feature)
        imgPaths.append([os.path.join(test_path, f) for f in os.listdir(test_path)])
        for imgpath in os.listdir(test_path):
            print(imgpath)
            PIL_img = Image.open(os.path.join(test_path,imgpath)).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            FaceSamples.append(img_numpy)
            names.append(feature)
            
            test_id = -1
            for i in range(len(name2id)):
                if name2id[str(i)] == feature:
                    test_id = i
                    break
            print(i)
            ids.append(int(test_id))
    
    #print(Features)
    #print(imgPaths) 
    return FaceSamples, ids, names
    
 
    
# 训练
def FaceTrain():
    print ("Train Face .... ")
    
    faces,ids,names=GetFeatureAndLabels('FaceData')
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))   #开始训练
    print(recognizer)
    
    recognizer.write(r'FaceDataConfig/trainer.yml')

    print("{0} faces is trained".format(len(np.unique(ids))))
    Say('训练完成')


def FaceDetection():
    register_status_init() #初始化签到信息表
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(facexml) # 加载人脸特征库
    
    recoginzer = cv2.face.LBPHFaceRecognizer_create()
    recoginzer.read(r'FaceDataConfig/trainer.yml')  #加载分类器
    
    name2id=json_read()

    
   # count=len(os.listdir(r'./FaceData/unknown'))
   # print(count)
    name = set([])
    register_status = json_read(JsonPath=r"FaceDataConfig/status.json")
    while(True):
        ret, frame = cap.read() # 读取一帧的图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 转灰
    
        faces = face_cascade.detectMultiScale(gray, 
            scaleFactor = 1.15, 
            minNeighbors = 5, 
            minSize = (5, 5)
            ) # 检测人脸

        
        Area=0.0
        test_img=[]
        xx,yy,ww,hh = 0,0,0,0
        for(x, y, w, h) in faces:
            if (w*h > Area and float(w + h) > 300.0):
                test_img=gray[y:y+h, x:x+w]
                xx,yy,ww,hh = x,y,w,h
                Area = w*h
            
        conf=int(0)
        test_name = set([])
        if len(test_img) != 0:
            plt.imshow(test_img)
            plt.show()
            cv2.rectangle(frame, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2) # 用矩形圈出人脸
            idnum, confidence = recoginzer.predict(gray[y:y+h, x:x+w])
            print ("id = ", idnum, "confidence = ", confidence)
            conf = 100 - confidence
            if confidence < 100:
                idnum = name2id[str(idnum)]
                confidence = "{0}%".format(round(100-confidence))
            else:
                idnum = "unknown"
                confidence = "{0}%".format(round(100-confidence))
            cv2.putText(frame, str(idnum), (xx+5, yy-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)
            cv2.putText(frame, str(confidence), (xx+5, yy+hh-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)
            
            test_name.add(idnum)
        
        print("Number of face: ", len(faces))
        cv2.imshow('video', frame)
        print(conf)
        
        for tname in test_name:
            if tname not in name:
                if tname == "unknown":
                    Say('你好,你是谁？')
                else:
                    if conf >= 45:
                        Say('你好,'+tname+',签到成功！')
                        name.add(tname)
                        register_status[tname] = "True"
                        json_write(register_status,JsonPath=r"FaceDataConfig/status.json")
                    else:
                        pass
        
        #if count == 100:
            #break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release() # 释放摄像头
    cv2.destroyAllWindows()
    


        
def main():
    Say("准备环境,欢迎使用myface签到系统")
    if first == 'y':
        Say("环境准备成功")
    
    while (True):
        SaySth("请选择操作,,0,开始签到,,1,管理员模式,,q,离开系统")
        
        order = input("\n please input a order(0(register),1(adminstrator),q(exit))")
        if order == "q":
            SaySth("谢谢使用, 再见！")
            break
        
        elif order == "0":   
            print(os.path.exists('./FaceDataConfig/trainer.yml'))
            if (os.path.exists('./FaceDataConfig/trainer.yml')):# 判断是否存在目录
                if (os.path.exists('FaceDataConfig/name.json')):# 判断是否存在目录
                    Say("开始检测")
                    FaceDetection()
                else:
                    Say("没有找到name.json文件,请联系管理员")
                    continue   
            else:
                Say("没有找到yml文件,请联系管理员")
                continue

        elif order == "1":
            SaySth("欢迎来到管理员0系统,,t,训练,,g,获取脸部特征")
            mod = input("\n please input a mod, t(training), g(get face)")
            if mod == "t":
                if (os.path.exists('FaceData') & (int)(len([os.listdir('FaceData')])) >= 1):# 判断是否存在目录
                    if (os.path.exists('FaceDataConfig/name.json')):# 判断是否存在目录
                        Say("训练中")
                        FaceTrain() 
                    else:
                         Say("没有找到name.json文件,请联系管理员")
                         continue 
                else:
                    Say("没有找到脸部数据,请联系管理员")
                    continue
            elif mod == "g":
                GetFeature()
                FaceTrain()
         

if __name__ == '__main__':
    main()
    