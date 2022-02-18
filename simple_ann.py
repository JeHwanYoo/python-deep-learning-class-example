import numpy as np

class ANN: 
  def __init__(self, layers):
    self.W = []
    self.b = []
    for x_nodes, h_nodes in layers:
      self.add_layer(x_nodes, h_nodes)
  
  def add_layer(self, x_nodes, h_nodes):
    self.W.append(np.random.rand(x_nodes, h_nodes))
    self.b.append(np.random.rand(h_nodes))

  def feed_forward(self):
    delta = 1e-7
    a = self.x
    for i, w in enumerate(self.W):
      b = self.b[i]
      z = np.dot(a, w) + b
      a = self.activate(z)
    y = a
    return -np.sum(self.t*np.log(y + delta) + (1-self.t) * np.log((1 - y) + delta))
  
  def train(self, x, t, epochs=1, learning_rate=1e-2, activate=None, debug_step=None):
    self.x = x
    self.t = t
    self.activate = activate or ANN.sigmoid
    if not debug_step:
        debug_step = int(epochs * 0.1)
    f = lambda x: self.feed_forward()
    for step in range(epochs):
      for i in range(len(self.W)):
        self.W[i] -= learning_rate * ANN.derivative(f, self.W[i])
        self.b[i] -= learning_rate * ANN.derivative(f, self.b[i])
      if step % debug_step == 0:
        print('step = ', step, 'loss value = ', self.feed_forward())

  def predict(self, input_data):
    a = input_data
    for i, w in enumerate(self.W):
      b = self.b[i]
      z = np.dot(a, w) + b
      a = self.activate(z)
    return a

  @staticmethod
  def derivative(f, x, dx=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + dx
        fx1 = f(x)
        x[idx] = tmp_val - dx
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2 * dx)
        x[idx] = tmp_val
        it.iternext()
    return grad

  @staticmethod
  def sigmoid(z):
    return 1 / (1+np.exp(z))

  # XOR Problem Example
  if __name__ == '__main__':
    xdata = np.array([[0, 0],[0, 1],[1, 0],[1, 1]]).reshape(4,2)
    tdata = np.array([0, 1, 1, 0]).reshape(4,1)

    xor_model = ANN([(2, 2), (2, 1)])
    xor_model.train(xdata, tdata, epochs=30001)

    test_data = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])

    for x in test_data:
      y = xor_model.predict(x)
      if y >= 0.5:
        print(f'{x} = 1')
      else:
        print(f'{x} = 0')
