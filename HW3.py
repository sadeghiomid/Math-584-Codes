import numpy as np 
import matplotlib.pyplot as plt

#Part I
def qrfactor(A3):
    A=np.zeros(np.shape(A3))
    m,n=np.shape(A)
    for i in range(n):
        A[:,i]=A3[:,i]

    Q=np.identity(m)
    for k in range(n):
        z=A[k:,k]
        v=np.append(-1*np.sign(z[0])*np.linalg.norm(z)-z[0],-z[1:])
        v=v/np.sqrt(np.vdot(v,v))
        for j in range(n):
            A[k:,j]=A[k:,j]-2*np.vdot(v,A[k:,j])*v

        for j in range(m):
            Q[k:,j]=Q[k:,j]-2*np.vdot(v,Q[k:,j])*v

    Q=Q.T 
    R=np.triu(A)
    return Q,R

def mgs(A):
    m,n=np.shape(A)
    V=np.zeros((m,n))
    Q=np.zeros((m,n))
    R=np.zeros((n,n))
    for i in range(n):
        V[:,i]=A[:,i]
        
    for j in range(n):
        R[j,j]=np.linalg.norm(V[:,j])
        Q[:,j]=V[:,j]/R[j,j]
        for k in range(j+1,n):
            R[j,k]=np.vdot(Q[:,j],A[:,k])
            V[:,k]=V[:,k]-R[j,k]*Q[:,j]

    return Q,R

m=20
n=19
err_qrfactor=np.zeros(50)
err_mgs=np.zeros(50)
err_qr=np.zeros(50)
A=np.random.normal(size=(m,n))
A2=np.zeros((m,n+1))
A2[:,:n]=A
A2[:,-1]=A[:,0]
noise=1e-10
nv=np.random.uniform(0,1,m)
for j in range(50):
    A2[:,-1]+=noise*nv
    q1,r1=qrfactor(A2)
    err_qrfactor[j]=np.linalg.norm(np.matmul(q1,r1)-A2)/np.linalg.norm(A2)
    q2,r2=mgs(A2)
    err_mgs[j]=np.linalg.norm(np.matmul(q2,r2)-A2)/np.linalg.norm(A2)
    q3,r3=np.linalg.qr(A2)
    err_qr[j]=np.linalg.norm(np.matmul(q3,r3)-A2)/np.linalg.norm(A2)
   
plt.plot(range(50),err_qrfactor,color='red')
plt.plot(range(50),err_mgs,color='blue')
plt.plot(range(50),err_qr,color='green')
plt.title('qrfactor vs. mgs vs. qr')
plt.xlabel('j')
plt.ylabel('Relative error')
plt.legend(['qrfactor','mgs','qr'],loc=0)
plt.show()

#Part II
dx=0.001
x=np.arange(1.920,2.081,dx)
px1=x**9 - 18*x**8 + 144*x**7 - 672*x**6 +2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512
px2=(x-2)**9
plt.plot(x,px1)
plt.plot(x,px2)
plt.title('Part II')
plt.xlabel('$x$')
plt.ylabel('$p(x)$')
plt.legend(['RHS','LHS'])
plt.show()
#Part III
#(a)
n=20
tp=np.zeros(100-n)
for m in range(n+1,101):
    tp[m-n-1]=np.linalg.cond(np.random.normal(size=(m,n)))

plt.figure()
plt.plot(range(n+1,101),tp)
plt.title('Condition number for $n=20$')
plt.xlabel('$m$')
plt.ylabel('Condition number')
m=100
tp2=np.zeros(80)
for n in range(20,m):
    tp2[n-20]=np.linalg.cond(np.random.normal(size=(m,n)))

plt.figure()
plt.plot(range(20,m),tp2)
plt.title('Condition number for $m=100$')
plt.xlabel('$n$')
plt.ylabel('Condition number')
#(b)
m2=20
n2=19
A=np.random.normal(size=(m2,n2))
A2=np.zeros((m2,n2+1))
A2[:,:n2]=A
A2[:,-1]=A[:,0]
cdn=np.linalg.cond(A2)
dt=np.linalg.det(A2)
#(c)
cdn2=np.zeros(5)
noise=1e-10
nv=np.random.uniform(0,1,m2)
for j in range(5):
    A2[:,-1]+=noise*nv
    cdn2[j]=np.linalg.cond(A2)

plt.plot([0,1,2,3,4,5],np.log(np.append(cdn,cdn2)))
plt.xlabel('$\epsilon$')
plt.ylabel('Log of condition number')
plt.title('Effect of noise on condition number')
plt.xticks(ticks=[0,1,2,3,4,5],labels=('0','1e-10','2e-10','3e-10','4e-10','5e-10'))