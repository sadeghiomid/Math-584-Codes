import os,sys
import imageio
import numpy as np
import matplotlib.pyplot as plt

#Part 1a
A1=np.random.normal(size=(10,10))
A=(A1+A1.T)/2
evalue,evector=np.linalg.eig(A)
plt.scatter(range(10),np.sort(evalue))
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.title('Spectrum of the random symmetric matrix A (part I(a))')
#Part 1b
def PI(A,T,v):
#    v=np.random.normal(size=(A.shape[1],1))
    v=v/np.linalg.norm(v)
    for t in range(T):
        tp=np.matmul(A,v)
        tp=tp/np.linalg.norm(tp)
        v=tp
    
    return np.matmul(np.matmul(np.conj(v).T,A),v),v

T=np.linspace(10,1000,100)
evalue_acc=np.zeros(100)
evector_acc=np.zeros(100)
v=np.random.normal(size=(A.shape[1],1))
counter=0
i=np.argmax(abs(evalue))
vi=evector[:,i]
for t in T:
    tp1,tp2=PI(A,int(t),v)
    evalue_acc[counter]=abs(tp1-evalue[i])/abs(evalue[i])
    evector_acc[counter]=np.linalg.norm(tp2-vi,np.inf)/np.linalg.norm(vi,np.inf)
plt.plot(T,evalue_acc)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Eigenvalue accuracy (Part I(b))')
plt.plot(T,evector_acc)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Eigenvector accuracy (Part I(b))')

#Part 1c
def RQI(A,eps,v0):
    v=v0/np.linalg.norm(v0)
    l=np.zeros(2,dtype=complex)
    l[0]=np.matmul(np.matmul(np.conj(v).T,A),v)
    tp=np.linalg.solve((A-l[0]*np.eye(A.shape[0])),v)
    tp=tp/np.linalg.norm(tp)
    v=tp
    l[1]=np.matmul(np.matmul(np.conj(v).T,A),v)
    while np.linalg.norm(l[1]-l[0])>eps:
        l[0]=l[1]
        tp=np.linalg.solve((A-l[0]*np.eye(A.shape[0])),v)
        tp=tp/np.linalg.norm(tp)
        v=tp
        l[1]=np.matmul(np.matmul(np.conj(v).T,A),v)
    
    return np.matmul(np.matmul(np.conj(v).T,A),v),v

def RQI2(A,T,v0):
    v=v0/np.linalg.norm(v0)
    l=np.zeros(2,dtype=complex)
    l[0]=np.matmul(np.matmul(np.conj(v).T,A),v)
    tp=np.linalg.solve((A-l[0]*np.eye(A.shape[0])),v)
    tp=tp/np.linalg.norm(tp)
    v=tp
    l[1]=np.matmul(np.matmul(np.conj(v).T,A),v)
    for t in range(T):
        l[0]=l[1]
        try:
            tp=np.linalg.solve((A-l[0]*np.eye(A.shape[0])),v)
        except np.linalg.LinAlgError:
            return np.matmul(np.matmul(np.conj(v).T,A),v),v

        tp=tp/np.linalg.norm(tp)
        v=tp
        l[1]=np.matmul(np.matmul(np.conj(v).T,A),v)
    
    return np.matmul(np.matmul(np.conj(v).T,A),v),v

#Method (i)
T=np.linspace(1,100,100)
evalue_acc=np.zeros((100,10))
for i in range(10):
    counter=0
    for t in T:
        [s,v2]=RQI2(A,int(t),evector[:,i]+0.05*np.random.normal(10))
        evalue_acc[counter,i]=abs(s-evalue[i])/abs(evalue[i])
        counter+=1

#Method (ii)
tt=np.linspace(-np.max(abs(evalue)),np.max(abs(evalue)),num=20)
sp=np.zeros((100,20))
T=np.linspace(1,100,100)
evalue_acc=np.zeros((100,10))
counter=0
for t in tt:
    counter2=0
    [ss,v]=PI(np.linalg.inv(A-t*np.eye(A.shape[0])),1,np.random.normal(size=(A.shape[1],1)))
    for T2 in T:
        [s,v2]=RQI2(A,int(T2),v)
        sp[counter2,counter]=np.real(s)
        counter2=counter2+1
    
    counter=counter+1
#Part 1d
evalue,evector=np.linalg.eig(A1)
s=np.linspace(10,1000,100,dtype=complex)
counter=0
for t in s:
    s[counter],v=PI(A1,int(np.real(t)),np.random.normal(size=(A1.shape[1],1))+1j*np.random.normal(size=(A1.shape[1],1)))
    counter+=1

x=[l.real for l in s]
y=[l.imag for l in s]
plt.scatter(x,y,color='red')
plt.scatter([evalue[1].real,evalue[2].real],[evalue[1].imag,evalue[2].imag],color='blue')
plt.legend(['Output of Power Iteration','True value'])
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Largest eigenvalue of non-symmetric matrix (Part I(d))')
#Part 2a
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
X=np.zeros((192*168,len(files)))
for t in range(len(files)):
    X[:,t]=np.reshape(imageio.imread(files[t]),192*168)

X=X-np.tile(X.mean(1),(2432,1)).T
[landa,v2]=PI(np.matmul(X.T,X),1000,np.random.normal(size=(2432,1)))
[u,s,v]=np.linalg.svd(X,full_matrices=False)
#Part 2b
[m,n]=X.shape
k=50
omega=np.random.normal(size=(n,k))
Y=np.matmul(X,omega)
[Q,R]=np.linalg.qr(Y)
B=np.matmul(Q.T,X)
[U,S,V]=np.linalg.svd(B,full_matrices=False)
Uapprox=np.matmul(Q,U)
#Part 2c
K=[5,10,15,20,25]
acc=np.zeros(5)
k=K[0]
omega=np.random.normal(size=(n,k))
Y=np.matmul(X,omega)
[Q,R]=np.linalg.qr(Y)
B=np.matmul(Q.T,X)
[U,S,V]=np.linalg.svd(B,full_matrices=False)
Uapprox=np.matmul(Q,U)
acc[0]=np.linalg.norm(S[0]*np.outer(Uapprox[:,0],V[0,:])-s[0]*np.outer(u[:,0],v[0,:]))/np.linalg.norm(s[0]*np.outer(u[:,0],v[0,:]))
plt.scatter(range(np.size(S)),S,color='b',marker='x')

k=K[1]
omega=np.random.normal(size=(n,k))
Y=np.matmul(X,omega)
[Q,R]=np.linalg.qr(Y)
B=np.matmul(Q.T,X)
[U,S,V]=np.linalg.svd(B,full_matrices=False)
Uapprox=np.matmul(Q,U)
acc[1]=np.linalg.norm(S[0]*np.outer(Uapprox[:,0],V[0,:])-s[0]*np.outer(u[:,0],v[0,:]))/np.linalg.norm(s[0]*np.outer(u[:,0],v[0,:]))
plt.scatter(range(np.size(S)),S,color='g',marker='x')

k=K[2]
omega=np.random.normal(size=(n,k))
Y=np.matmul(X,omega)
[Q,R]=np.linalg.qr(Y)
B=np.matmul(Q.T,X)
[U,S,V]=np.linalg.svd(B,full_matrices=False)
Uapprox=np.matmul(Q,U)
acc[2]=np.linalg.norm(S[0]*np.outer(Uapprox[:,0],V[0,:])-s[0]*np.outer(u[:,0],v[0,:]))/np.linalg.norm(s[0]*np.outer(u[:,0],v[0,:]))
plt.scatter(range(np.size(S)),S,color='k',marker='x')

k=K[3]
omega=np.random.normal(size=(n,k))
Y=np.matmul(X,omega)
[Q,R]=np.linalg.qr(Y)
B=np.matmul(Q.T,X)
[U,S,V]=np.linalg.svd(B,full_matrices=False)
Uapprox=np.matmul(Q,U)
acc[3]=np.linalg.norm(S[0]*np.outer(Uapprox[:,0],V[0,:])-s[0]*np.outer(u[:,0],v[0,:]))/np.linalg.norm(s[0]*np.outer(u[:,0],v[0,:]))
plt.scatter(range(np.size(S)),S,color='m',marker='x')

k=K[4]
omega=np.random.normal(size=(n,k))
Y=np.matmul(X,omega)
[Q,R]=np.linalg.qr(Y)
B=np.matmul(Q.T,X)
[U,S,V]=np.linalg.svd(B,full_matrices=False)
Uapprox=np.matmul(Q,U)
acc[4]=np.linalg.norm(S[0]*np.outer(Uapprox[:,0],V[0,:])-s[0]*np.outer(u[:,0],v[0,:]))/np.linalg.norm(s[0]*np.outer(u[:,0],v[0,:]))
plt.scatter(range(np.size(S)),S,color='c',marker='x')

plt.scatter(range(25),s[:25],color='red',marker='.')
plt.legend(['k=5','k=10','k=15','k=20','k=25','True singular values'])
plt.title('Singular values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.scatter(K,acc)
plt.title('Relative error of the norm of the leading mode')
plt.xlabel('k')
plt.xticks([5,10,15,20,25])
plt.ylabel('Relative error')