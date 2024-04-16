import os

all_videos={}
with open("/data3/ava_file_names_trainval_v2.1.txt","r") as f:
    data = f.readlines()
    print(type(data))
    for i in data:
        all_videos[i]=1

print(len(all_videos))


my_videos={}
with open("/data3/video.txt","r") as f:
    data = f.readlines()
    for i in data:
        my_videos[i]=1

print(len(my_videos))

print(all_videos.items()-my_videos.items())
