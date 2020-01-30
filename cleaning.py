import os
for dirname in os.listdir("/Users/aman/Face_recognition/data/"):
    if os.path.isdir(dirname):
        for i, filename in enumerate(os.listdir(dirname)):
            os.rename(dirname + "/" + filename, dirname + "/" + "Aman" + str(i) + ".jpeg")
            #print(dirname,filename)
