import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def train_model(data, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    return model, score

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Number of iterations
num_iterations = 100

accuracies = []

for i in range(num_iterations):
    print(f"Iteration {i+1}:")
    model, score = train_model(data, labels)
    print('Accuracy: {}%'.format(score * 100))
    accuracies.append(score)

average_accuracy = np.mean(accuracies)
print('Average Accuracy: {}%'.format(average_accuracy * 100))

# Save the last trained model
with open('final_model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
