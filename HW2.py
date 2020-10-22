import os,sys
import imageio
import numpy as np
import matplotlib.pyplot as plt

path="C:\\Users\\sadeg\\Downloads\\yalefaces_cropped\\CroppedYale"
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

files=[]
files=getListOfFiles(path)
A=np.zeros((192*168,len(files)))
for t in range(len(files)):
    A[:,t]=np.reshape(imageio.imread(files[t]),192*168)

A=A-np.tile(A.mean(1),(2432,1)).T
#Part 1
u,s,v=np.linalg.svd(A,full_matrices=False)
#Part 2
plt.imshow(np.reshape(u[:,0],(192,168)))
plt.title('First eigenface')
plt.imshow(np.reshape(u[:,1],(192,168)))
plt.title('Second eigenface')
plt.imshow(np.reshape(u[:,2],(192,168)))
plt.title('Third eigenface')
plt.imshow(np.reshape(u[:,3],(192,168)))
plt.title('Fourth eigenface')
plt.imshow(np.reshape(u[:,4],(192,168)))
plt.title('Fifth eigenface')
#Part 3
uu,ss,vv=np.linalg.svd(A[:,:2304],full_matrices=False)
plt.scatter(range(len(ss)),ss)
plt.title('Singular value spectrum of cropped images')
plt.xlabel('i')
plt.ylabel('Singular value, $\sigma_i$')
plt.show()
plt.imshow(np.reshape(np.matmul(np.matmul(uu[:,:40],uu[:,:40].T),A[:,2304]),(192,168)))
#Part 4
path2="C:\\Users\\sadeg\\Downloads\\yalefaces_uncropped\\yalefaces"
files2=getListOfFiles(path2)
A2=np.zeros((243*320,len(files2)))
for t in range(len(files2)):
    A2[:,t]=np.reshape(imageio.imread(files2[t]),243*320)

A2=A2-np.tile(A2.mean(1),(165,1)).T
u2,s2,v2=np.linalg.svd(A2[:,:150],full_matrices=False)
plt.scatter(range(len(s2)),s2)
plt.title('Singular value spectrum of uncropped images')
plt.xlabel('i')
plt.ylabel('Singular value, $\sigma_i$')
plt.show()
plt.imshow(np.reshape(np.matmul(np.matmul(u2[:,:40],u2[:,:40].T),A2[:,160]),(243,320)))

