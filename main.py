import face_recognition as fr
import numpy as np
import pandas as pd
from os import listdir,mkdir
import shutil
from PIL import Image
from random import shuffle
from functools import reduce
from csv import writer

def init(fpath="images"):
    mkdir(f'{fpath}/grouped')
    mkdir(f'{fpath}/processed')
    mkdir(f'{fpath}/uniquefaces')

def findUniqueFacesImage(ipath,pad_images=True):
    image = fr.load_image_file(ipath)
    faceLocs = fr.face_locations(image)
    faces = []
    for (t,r,b,l) in faceLocs:
        if pad_images: padw,padh = (r-l)//4,(b-t)//2
        else: padw,padh = 0,0
        faces.append(image[t-padh:b+padh,l-padw:r+padw])
    return faces

def findUniqueFaces(fpath="images/new",sample_size=2000,shuffle_files=True,show_progress=True):
    uniqueFaces = []
    uniqueFaceAverages = []

    files = listdir(fpath)
    if shuffle_files: shuffle(files)

    count = 1
    for i in files[:sample_size]:
        faces = findUniqueFacesImage(f'{fpath}/{i}')
        for face in faces:
            try: encoding = fr.face_encodings(face)[0]
            except:
                print(i)
                continue
            
            closestDist = 10
            closestImage = 0
            found = 1
            for groupIndex in range(len(uniqueFaces)):
                refEncoding = uniqueFaceAverages[groupIndex]
                comp = np.linalg.norm(refEncoding-encoding)
                if comp <= 0.15:
                    refEncoding *= len(uniqueFaces[groupIndex])
                    refEncoding += encoding
                    refEncoding /= len(uniqueFaces[groupIndex]) + 1
                    uniqueFaces[groupIndex].append(face)
                    found = 2
                    break
                if comp <= 0.4 and comp < closestDist:
                    found = 3
                    closestDist = comp
                    closestImage = (encoding,groupIndex)
            if found == 2:
                continue
            if found == 1 or closestDist == 10:
                uniqueFaces.append([face])
                uniqueFaceAverages.append(encoding)
                continue
            if found == 3:
                encoding, groupIndex = closestImage
                refEncoding = uniqueFaceAverages[groupIndex]
                refEncoding *= len(uniqueFaces[groupIndex])
                refEncoding += encoding
                refEncoding /= len(uniqueFaces[groupIndex]) + 1
                uniqueFaces[groupIndex].append(face)
        if show_progress: print(count)
        count+=1
    return(uniqueFaces)

def saveUniques(uniqueFaces):
    if uniqueFaces == 0:
        uniqueFaces = findUniqueFaces()
    uniqueFaces.sort(key=len,reverse=True)
    mkdir('images/uniquefaces/misc')
    for i in range(len(uniqueFaces)):
        group = uniqueFaces[i]
        misc = len(group) == 1
        if not misc: mkdir(f'images/uniquefaces/{i}')
        for j in range(len(group)):
            Image.fromarray(group[j]).save(f'images/uniquefaces/misc/{i}_{j}.png' if misc else f'images/uniquefaces/{i}/{j}.png')
    return(len(listdir('images/uniquefaces')))

def readFacesAvg(fpath="images/uniquefaces",show_progress=True):
    avgFaces = []
    count = 1
    for i in listdir(fpath):
        if i=="misc": continue
        if "_" not in i: continue
        name = i.split('_')[0]
        rawencodings = [fr.face_encodings(fr.load_image_file(f'{fpath}/{i}/{j}')) for j in listdir(f'{fpath}/{i}')]
        encodings = [encoding[0] for encoding in rawencodings if len(encoding)!=0]
        if len(encodings) == 0: continue
        avgFaces.append({"name":name,"face":reduce(lambda a,b:a+b, encodings)/len(encodings)})
        if show_progress: print(count)
        count+=1
    return(avgFaces)

def findFacesImage(ipath,refs,tolerance=0.4):
    encodings = fr.face_encodings(fr.load_image_file(ipath))
    foundFaces = list(set([ref["name"] for ref in refs if True in fr.compare_faces(encodings, ref["face"], tolerance=tolerance)]))
    return([len(encodings),"|".join(foundFaces) if len(foundFaces)!=0 else 0,len(foundFaces)])

def findFacesFolder(fpath="images",show_progress=True):
    refs = readFacesAvg()
    with open("result.csv","w") as fh:
        fh.write("filename,numOfFaces,foundFaces,numOfFound\n")
    writer_object = writer(open("result.csv","a"))
    count = 1
    for i in listdir(f'{fpath}/new'):
        writer_object.writerow([i]+findFacesImage(f'{fpath}/new/{i}',refs,tolerance=0.4))
        shutil.move(f'{fpath}/new/{i}',f'{fpath}/processed/{i}')
        if show_progress: print(count)
        count+=1
    fh.close()

def resetImages(fpath="images"):
    for i in listdir(f'{fpath}/processed'):
        shutil.move(f'{fpath}/processed/{i}',f'{fpath}/new/{i}')

def readImageGroups():
    groups_df = pd.read_csv("groups.csv")
    groups = {}

    groups = {group.groupname:group.names.split("|") for _,group in groups_df.iterrows()}
    groups_df["names"] = groups_df["names"].str.split("|")
    groups_df = groups_df.explode("names")

    result_df = pd.read_csv("result.csv")[["filename","foundFaces"]]
    result_df["foundFaces"] = result_df["foundFaces"].str.split("|")
    result_df = result_df.explode("foundFaces")
    nonefound = result_df.loc[result_df["foundFaces"]=="0"]["filename"]
    result_df = result_df.loc[result_df["foundFaces"]!="0"]
    result_df.columns = ["filename","names"]

    imagegroup_df = groups_df.merge(result_df,on="names",how="inner")
    return(imagegroup_df,groups,nonefound)

def createGroupFolders():
    imagegroup_df,groups,nonefound = readImageGroups()
    groupnames = list(set(imagegroup_df["groupname"]))
    mkdir('images/grouped/misc')
    for filename in nonefound:
        shutil.copy(f'images/processed/{filename}',f'images/grouped/misc/{filename}')
    for group in groupnames:
        groupfiles = list(set(imagegroup_df.loc[imagegroup_df["groupname"]==group]["filename"]))
        mkdir(f'images/grouped/{group}')
        for filename in groupfiles:
            shutil.copy(f'images/processed/{filename}',f'images/grouped/{group}/{filename}')

        mkdir(f'images/grouped/{group}/only')
        groupmembers = groups[group]
        for filename in groupfiles:
            if all([name in groupmembers for name in list(set(imagegroup_df.loc[imagegroup_df["filename"]==filename]["names"]))]):
                shutil.copy(f'images/processed/{filename}',f'images/grouped/{group}/only/{filename}')