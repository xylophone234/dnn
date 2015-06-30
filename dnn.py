import numpy as np
import math

# trainSet=[[0.1,0.1,-0.9],[0.9,0.9,-0.9],[0.1,0.9,0.9],[0.9,0.1,0.9]]
trainSet=[[-1.0,-1.0,-1.0],[-1.0,1.0,1.0],[1.0,-1.0,1.0],[1.0,1.0,-1.0]]
trainSet=np.array(trainSet)
trainSet=np.transpose(trainSet)

def sigmoid(x):
	return 1.0/(1.0+math.exp(-x))
	# return math.tanh(x)
def dsigmoid(y):
	return y*(1.0-y)
	# return 1.0-y**2

def tanz(x):
	return math.tanh(x)
def dtanz(y):
	return 1.0-y*y

def relu(x):
	return max(0,x)
def drelu(x):
	if x>0 :
		return 1
	else:
		return 0

def ta(x):
	return 1.7159*math.tanh(2.0*x/3.0)
def dta(y):
	return 1.144*(1.0-y*y/2.9443)

def id(x):
	return x
def did(y):
	return 1.0

def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:  
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2

def dsoftmax(y):
	pass

# sigmoid_ufunc=np.vectorize(sigmoid,otypes=[np.float])
# dsigmoid_ufunc=np.vectorize(dsigmoid,otypes=[np.float])
# sigmoid_ufunc=np.vectorize(relu,otypes=[np.float])
# dsigmoid_ufunc=np.vectorize(drelu,otypes=[np.float])
# sigmoid_ufunc=np.vectorize(tanz,otypes=[np.float])
# dsigmoid_ufunc=np.vectorize(dtanz,otypes=[np.float])
sigmoid_ufunc=np.vectorize(ta,otypes=[np.float])
dsigmoid_ufunc=np.vectorize(dta,otypes=[np.float])

active={
	'sigmoid':{
		'a':np.vectorize(sigmoid,otypes=[np.float]),
		'd':np.vectorize(dsigmoid,otypes=[np.float])
	},
	'tanh':{
		'a':np.vectorize(tanz,otypes=[np.float]),
		'd':np.vectorize(dtanz,otypes=[np.float])
	},
	'relu':{
		'a':np.vectorize(relu,otypes=[np.float]),
		'd':np.vectorize(drelu,otypes=[np.float])
	},
	'ta':{
		'a':np.vectorize(ta,otypes=[np.float]),
		'd':np.vectorize(dta,otypes=[np.float])
	},
	'id':{
		'a':np.vectorize(id,otypes=[np.float]),
		'd':np.vectorize(did,otypes=[np.float])
	},
	'softmax':{
		'a':softmax,
		'd':dsoftmax
	}
}

class Layer:
	def __init__(self,n_input,n_output,sig='relu'):
		self.n_input=n_input
		self.n_output=n_output
		# self.w=np.random.uniform(low=-0.1,high=0.1,size=[n_output,n_input])
		self.w=np.random.randn(n_output,n_input)/math.sqrt(n_input)
		self.b=(np.zeros(n_output)).reshape(-1,1)
		self.z=None
		self.input=None
		self.output=None
		self.delta=None
		self.deltaw=None
		self.deltab=None
		self.lastDeltaw=np.zeros((n_output,n_input))
		self.lastDeltab=np.zeros((n_output,)).reshape(-1,1)
		self.a=active[sig]['a']
		self.d=active[sig]['d']



	def forward(self,x):
		self.input=x
		out=np.dot(self.w,self.input)
		self.z=out+self.b
		self.output=self.a(self.z)
		# print('w=',self.w)
		# print('b=',self.b)
		# print('input=,',self.input)
		# print('output=,',self.output)
		return self.output

	def adjust(self,delta,alpha,m):
		# print('delta',delta)
		self.delta=delta
		# self.deltaw=np.dot(self.delta,np.transpose(self.output)).sum(axis=1).reshape(-1,1) 
		self.deltaw=np.dot(self.delta,np.transpose(self.input))/self.input.shape[1]
		# self.deltaw=(self.delta*self.input).sum(axis=1).reshape(-1,1)
		self.deltab=self.delta.sum(axis=1).reshape(-1,1)/self.input.shape[1]
		# print (self.deltaw)
		self.w=self.w+self.deltaw*alpha+m*self.lastDeltaw
		self.lastDeltaw=self.deltaw*alpha
		# print('b',self.b)
		# print('delta',self.deltab.sum(axis=1))
		self.b=self.b+self.deltab*alpha+m*self.lastDeltab
		self.lastDeltab=self.deltab

class DNN:
	def __init__(self,layerList):
		self.layers=[]
		for i in range(len(layerList)-1):
			self.layers.append(Layer(layerList[i][0],layerList[i+1][0],layerList[i][1]))
		

	def forward(self,x):
		o=self.layers[0].forward(x)
		for i in range(len(self.layers)-1):
			o=self.layers[i+1].forward(self.layers[i].output)
		return o

	def train(self,trainSetx,trainSety,n,npatch=1,alpha=0.05,m=0.3):
		patchx=[]
		patchy=[]
		# npatch=1
		patchsize=int(trainSetx.shape[1]/npatch)
		print patchsize
		for i in range(npatch):
			patchx.append(trainSetx[:,i*patchsize:(i+1)*patchsize])
			patchy.append(trainSety[:,i*patchsize:(i+1)*patchsize])
		for i in range(n):
			for k in range(npatch):
				o=self.forward(patchx[k])
				self.layers[-1].delta=(patchy[k]-o)*self.layers[-1].d(self.layers[-1].output)
				for j in range(0,len(self.layers)-1)[::-1]:
					self.layers[j].delta=np.dot(np.transpose(self.layers[j+1].w),self.layers[j+1].delta)*self.layers[j].d(self.layers[j].output)
				for j in range(0,len(self.layers))[::-1]:
					self.layers[j].adjust(self.layers[j].delta,alpha,m)
				# # print('o',o)
				# deltao=(patchy[k]-o)*dsigmoid_ufunc(self.outputLayer.output)
				# # print('deltao',deltao)
				# deltah1=np.dot(np.transpose(self.outputLayer.w),deltao)*dsigmoid_ufunc(self.hidden1.output)
				# deltah=np.dot(np.transpose(self.hidden1.w),deltah1)*dsigmoid_ufunc(self.hiddenLayer.output)
				# # print('deltah',deltah)
				# self.outputLayer.adjust(deltao,alpha,m)
				# self.hidden1.adjust(deltah1,alpha,m)
				# self.hiddenLayer.adjust(deltah,alpha,m)
				# # print(o)
				print(((patchy[k]-o)*(patchy[k]-o)).sum()/(o.shape[0]*o.shape[1]))
	# todo
	def pretrain(self,x):
		for i in range(len(self.layers)-1):
			templayer=Layer(self.layers[i].n_output,self.layers[i].n_input,'ta')

def test():
	x=np.random.rand(2,1000)
	y=np.cos(x)
	dn=DNN([[2,'relu'],[10,'relu'],[2,'id']])
	dn.train(x,y,100,100)
	print x-dn.forward(x)
	t=np.random.rand(2,100000)
	e=np.cos(t)-dn.forward(t)
	print e
	print math.sqrt((e*e).sum()/100000)

if __name__ == '__main__':
	test()

