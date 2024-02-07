import panda as pd



myFile= '/home/40008593/Documents/M1/S2/Projet-HP/CleanTrain.csv'
donnees = pd.read_csv(myFile, encoding='utf-8')

y = donnees.SalePrice

# Step 3: Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Instantiate the KNN Classifier
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Step 5: Train the model
knn_classifier.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = knn_classifier.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

