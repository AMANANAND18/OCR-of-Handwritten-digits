import numpy as np
import cv2

# Read the image
image = cv2.imread('digits1.png')

# Convert to grayscale
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Split the image into 5000 small dimensions of size 20x20
divisions = [np.hsplit(row, 100) for row in np.vsplit(gray_img, 50)]
NP_array = np.array(divisions)

# Prepare training and test data
train_data = NP_array[:, :50].reshape(-1, 400).astype(np.float32)
test_data = NP_array[:, 50:100].reshape(-1, 400).astype(np.float32)

# Create labels for digits 0-9
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = train_labels.copy()

# Create kNN model
knn = cv2.ml.KNearest_create()

# Train the kNN model
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# Predict using the kNN model
ret, results, neighbours, dist = knn.findNearest(test_data, k=3)

# Check accuracy
matches = results == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / results.size

# Display accuracy
print(f'Accuracy: {accuracy}%')
