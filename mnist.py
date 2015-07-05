import numpy as np
import dnn

def loaddata():
	tr=np.loadtxt('../mnist/train.csv',dtype=np.str,delimiter=',')
	allt=tr[1:,1:].astype(np.float)
	allt=np.transpose(allt)/255
	trainx=allt[:,0:40000]
	valix=allt[:,40000:]
	z=tr[1:,0:1]
	y=np.zeros((42000,10))
	for i in range(42000):
		y[i,z[i,0]]=1
	y=np.transpose(y)
	trainy=y[:,0:40000]
	valiy=y[:,40000:]


	te=np.loadtxt('../mnist/test.csv',dtype=np.str,delimiter=',')
	allt=te[1:,:].astype(np.float)/255
	testx=np.transpose(allt)

	return trainx,trainy,valix,valiy,testx

def err(y,lable):
	e=y.argmax(axis=0)-lable.argmax(axis=0)
	n=len(e[e.nonzero()])
	print n
	return float(n)/y.shape[1]

print 'loading dataset'
trainx,trainy,valix,valiy,testx=loaddata()
print 'loading complete'
print 'start training'
dnlist=[]
result=0
valiresult=0
num=30
for i in range(num):
	print i
	dn=dnn.DNN([[784,'relu'],[30,'relu'],[20,'softmax'],[10]])
	dnlist.append(dn)
	errorrate=1.0
	while errorrate>0.1:
		dn.train(trainx,trainy,5,20,0.2,0.0)
		errorrate=err(dn.forward(valix),valiy)
		print 'errorrate=',errorrate
for i in range(num):
	valiresult=valiresult+dnlist[i].forward(valix)
	result=result+dnlist[i].forward(testx)
errorrate=err(valiresult,valiy)
print 'vali errorrate=',errorrate
result=result.argmax(axis=0).reshape((-1,1))
id=np.arange(1,28001).reshape((-1,1))
np.savetxt('result.csv',np.hstack((id,result)),fmt='%d',delimiter=',')
