import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Initialize RandomForestClassifier
model = RandomForestClassifier()

# Perform cross-validation
cv_scores = cross_val_score(model, data, labels, cv=5)  # 5-fold cross-validation
average_accuracy = np.mean(cv_scores)

print('Cross-Validation Mean Accuracy: {}%'.format(average_accuracy * 100))

# Train the final model on the entire dataset
final_model = RandomForestClassifier()
final_model.fit(data, labels)

# Save the final trained model
with open('final_model.p', 'wb') as f:
    pickle.dump({'model': final_model}, f)
