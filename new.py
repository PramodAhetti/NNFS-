import numpy as np

learning_rate = 0.001
df = 0.001

def train(weights):
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            prev_cost = cost()
            weights[i][j] += df
            after_cost = cost()
            weights[i][j] -= df
            if (after_cost - prev_cost) >= 0:
                weights[i][j] -= learning_rate
            else:
                weights[i][j] += learning_rate
    return weights

def calculate(input_layer, hidden_layer, output_layer, hidden_bias, output_bias):
    hidden_activation = (np.dot(input_layer, hidden_layer) + hidden_bias)**2
    output_activation = (np.dot(hidden_activation, output_layer) + output_bias)**2
    return output_activation

def cost():
    output = np.array([[1], [0], [0], [1]])
    acquired_output = calculate(input_layer, hidden_layer, output_layer, hidden_bias, output_bias)
    return np.sum((output - acquired_output) ** 2)

input_layer = np.array([[0, 0], 
                        [0, 1], 
                        [1, 0], 
                        [1, 1]])
hidden_layer = np.array([[0.1, 0.2], 
                         [0.3, 0.4]])
output_layer = np.array([[0.5], 
                         [0.6]])

hidden_bias = np.zeros((1, hidden_layer.shape[1]))
output_bias = np.zeros((1, output_layer.shape[1]))

def predict(input_layer, hidden_layer, output_layer, hidden_bias, output_bias):
    print(np.maximum(0, calculate(input_layer, hidden_layer, output_layer, hidden_bias, output_bias)))

print("Initial cost:", cost())

count = 0
epochs = 5000

while epochs:
     epochs -= 1
     print("Iteration:", count, cost())
     count += 1
     hidden_layer = train(hidden_layer)
     output_layer = train(output_layer)
     hidden_bias = train(hidden_bias)
     output_bias = train(output_bias)

print("After training:", cost())
predict(input_layer, hidden_layer, output_layer, hidden_bias, output_bias)
print("The equation for the trained model is:")
print(hidden_layer)
print(hidden_bias)
print(output_layer)
print(output_bias)
