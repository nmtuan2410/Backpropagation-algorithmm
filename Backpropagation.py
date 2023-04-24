
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist  
(x_train,y_train),(x_test,y_test)= mnist.load_data()
x_train=np.reshape(x_train,(60000,784))/255.0       
x_test= np.reshape(x_test,(10000,784))/255.0
y_train = np.matrix(np.eye(10)[y_train])
y_test = np.matrix(np.eye(10)[y_test])

def relu(x):  
  return(np.maximum(0,x))

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def softmax(x):
    return np.divide(np.matrix(np.exp(x)),np.mat(np.sum(np.exp(x),axis=1)))

def Forwardpass(X,Who,bho,Wh,bh,Wo,bo): 
    zho= X@Who.T + bho                    
    ao=relu(zho)
    zh = ao@Wh.T + bh
    a = sigmoid(zh)
    z=a@Wo.T + bo
    o = softmax(z)
    return o
def AccTest(label,prediction):   
    OutMaxArg=np.argmax(prediction,axis=1)
    LabelMaxArg=np.argmax(label,axis=1)
    Accuracy=np.mean(OutMaxArg==LabelMaxArg)
    return Accuracy

learningRate = 0.5
Epoch=50         
NumTrainSamples=60000
NumTestSamples=10000
NumInputs=784
NumHiddenUnits=512      
NumClasses=10           
#hidden layer1    
Who=np.matrix(np.random.uniform(-0.5,0.5,(512,NumInputs)))   
bho= np.random.uniform(0,0.5,(1,512))
dWho= np.zeros((NumHiddenUnits,NumInputs))
dbho= np.zeros((1,NumHiddenUnits))
#hidden layer2
Wh=np.matrix(np.random.uniform(-0.5,0.5,(NumHiddenUnits,512))) 
bh= np.random.uniform(0,0.5,(1,NumHiddenUnits))
dWh= np.zeros((NumHiddenUnits,512))
dbh= np.zeros((1,NumHiddenUnits))
#Output layer
Wo=np.random.uniform(-0.5,0.5,(NumClasses,NumHiddenUnits))  
bo= np.random.uniform(0,0.5,(1,NumClasses))
dWo= np.zeros((NumClasses,NumHiddenUnits))
dbo= np.zeros((1,NumClasses))
from IPython.display import clear_output
loss = []
Acc = []
Batch_size = 200   
Stochastic_samples = np.arange(NumTrainSamples)
for ep in range (Epoch):
  np.random.shuffle(Stochastic_samples)
  for ite in range (0,NumTrainSamples,Batch_size): 
    #feed forward propagation
    Batch_samples = Stochastic_samples[ite:ite+Batch_size]
    x = x_train[Batch_samples,:]
    y=y_train[Batch_samples,:]  
    zho= x@Who.T + bho          
    ao=relu(zho)               
    zh = ao@Wh.T + bh           
    a = relu(zh)
    z=a@Wo.T + bo
    o = softmax(z)
 
    loss.append(-np.sum(np.multiply(y,np.log10(o))))
    d = o-y   
    # backpropagation 
    dh = d@Wo   
    dhs = np.multiply(np.multiply(dh,a),(1-a))  #for sigmoid
    dho=dhs@Wh   
    dhr= np.multiply(np.ceil(ao/1+ao),dho) #for relu
    dWo = np.matmul(np.transpose(d),a)  
    dbo = np.mean(d)   
    dWh = np.matmul(np.transpose(dhs),ao)
    dbh = np.mean(dhs)  
    dWho = np.matmul(np.transpose(dhr),x)    
    dbho = np.mean(dhr)  
    Wo =Wo - learningRate*dWo/Batch_size          
    bo =bo - learningRate*dbo
    Wh =Wh-learningRate*dWh/Batch_size
    bh =bh-learningRate*dbh
    Who =Who-learningRate*dWho/Batch_size
    bho =bho-learningRate*dbho
 
  prediction = Forwardpass(x_test,Who,bho,Wh,bh,Wo,bo)   
  Acc.append(AccTest(y_test,prediction))                 
  print('Epoch:', ep )
  print('Accuracy:',AccTest(y_test,prediction) )

prediction = Forwardpass(x_test,Who,bho,Wh,bh,Wo,bo)
Rate = AccTest(y_test,prediction)
print(Rate)