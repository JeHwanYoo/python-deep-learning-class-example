import numpy as np

class ANN:
  def __init__(self, nodes):
    self.W = [None]
    self.b = [None]
    self.Z = []
    self.A = []
    self.N = len(nodes)

    for i in range(1, self.N):
      self.W.append(np.random.randn(nodes[i-1], nodes[i]) / np.sqrt(nodes[i-1]/2))
      self.b.append(np.random.rand(nodes[i]))

    for i in range(self.N):
      self.Z.append(np.zeros([1,nodes[i]]))
      self.A.append(np.zeros([1,nodes[i]]))


  def feed_forward(self):  
    delta = 1e-7
    self.Z[0] = self.input_data
    self.A[0] = self.input_data
    
    for i in range(1, self.N):
      self.Z[i] = np.dot(self.A[i-1], self.W[i]) + self.b[i]
      self.A[i] = self.activate(self.Z[i])
    y = self.A[-1]
    
    return -np.sum(self.target_data*np.log(y+delta)+(1-self.target_data)*np.log((1-y)+delta))     

  def loss_val(self):  
    delta = 1e-7
    self.Z[0] = self.input_data
    self.A[0] = self.input_data
    
    for i in range(1, self.N):
      self.Z[i] = np.dot(self.A[i-1], self.W[i]) + self.b[i]
      self.A[i] = self.activate(self.Z[i])
    y = self.A[-1]
    
    return -np.sum(self.target_data*np.log(y+delta)+(1-self.target_data)*np.log((1-y)+delta))    
   

  def train(self, input_data, target_data, learning_rate=1e-2, activate=None):
    self.input_data = input_data
    self.target_data = target_data    
    self.activate = activate or ANN.sigmoid
    
    # 먼저 feed forward 를 통해서 최종 출력값과 이를 바탕으로 현재의 에러 값 계산
    self.feed_forward()
    
    # 출력층 오차역전파
    loss = (self.A[-1]-self.target_data) * self.A[-1] * (1-self.A[-1])
    self.W[-1] = self.W[-1] - learning_rate * np.dot(self.A[-2].T, loss)   
    self.b[-1] = self.b[-1] - learning_rate * loss 

    # 은닉층 오차역전파
    for i in range(self.N-2, 0, -1):
      loss = np.dot(loss, self.W[i+1].T) * self.A[i] * (1-self.A[i])
      self.W[i] = self.W[i] - learning_rate * np.dot(self.A[i-1].T, loss)   
      self.b[i] = self.b[i] - learning_rate * loss

  def predict(self, input_data):
    a = input_data
    
    for i in range(1, self.N):
      z = np.dot(a, self.W[i]) + self.b[i]
      a = self.activate(z)
    y = a

    return np.argmax(y)

  def accuracy(self, test_input_data, test_target_data):
    matched = []
    unmatched = []
    
    for index in range(len(test_input_data)):
      label = int(test_target_data[index])
      data = (test_input_data[index] / 255.0 * 0.99) + 0.01
      predicted_num = self.predict(np.array(data, ndmin=2))
  
      if label == predicted_num:
        matched.append(index)
      else:
        unmatched.append(index)

    print("Current Accuracy = ", (len(matched)/(len(test_input_data))))
    
    return matched, unmatched

  @staticmethod
  def sigmoid(z):
    return 1/(1+np.exp(-z))
  
