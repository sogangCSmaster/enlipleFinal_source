from __future__ import print_function
import os
import pandas as pd
import sys

file_list=[]
foldername = './data'
for path, dirs, files in os.walk(foldername):
    if files:
        for filename in files:
            fullname = os.path.join(path, filename)
            file_list.append(fullname)

if not os.path.isdir('dataset'):
    os.mkdir('dataset')

print("File processing...")
for file in file_list:
    section = file.split('/')[1]
    classname = file.split('/')[2]
    subclass = file.split('/')[3].split('.')[0]
    
    df = pd.read_csv(file, encoding = 'utf-8')

    selected = ['title','body']
    title_raw = df[selected[0]].tolist()
    body_raw = df[selected[1]].tolist()
    for i in range(len(body_raw)):
        if not os.path.isdir('dataset/'+ section):
            os.mkdir('dataset/'+ section)
        if not os.path.isdir('dataset/'+'/'+ section +'/'+ classname ):
            os.mkdir('dataset/'+'/'+ section +'/'+ classname)
        if not os.path.isdir('dataset/'+'/'+ section +'/'+ classname +'/'+ subclass ):
            os.mkdir('dataset/'+'/'+ section +'/'+ classname +'/'+ subclass)
        fp = open('dataset/'+'/'+ section +'/'+ classname +'/'+ subclass + '/' + str(i)+'.txt','w', encoding = 'utf-8')
        fp.write(str(body_raw[i]))
        fp.close()
