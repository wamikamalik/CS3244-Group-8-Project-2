import os
import cv2
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import datasets, layers, models
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, get_scorer_names
from sklearn import metrics


def naming(path):
    return path.split("\\")[-1]


def trainortest(path):
    return "train" == path.split("\\")[-3]


def findclassindex(path):
    return path.split("\\")[-2][1]


datafolder = "Combined New"
destinationfolder = "Prediction New"
finalfolder = "Extracted New"

CLASS = [["c0", "Safe Driving"], ["c1", "Text"], ["c2", "Phone"],
           ["c3", "Adjusting Radio"], ["c4", "Drinking"],
           ["c5", "Reaching Behind"], ["c6", "Hair or Makeup"],
           ["c7", "Talking to Passenger"]]

TEST_CLS = [os.path.join(os.getcwd(), "Distracted Driver Dataset", datafolder, "test", cls[0]) for cls in CLASS]
TRAIN_CLS = [os.path.join(os.getcwd(), "Distracted Driver Dataset", datafolder, "train", cls[0]) for cls in CLASS]

X_train = []
X_test = []
Y_train = []
Y_test = []

for cls in TEST_CLS:
    for ori_name in os.listdir(cls):
        ori_img = cv2.imread(os.path.join(os.getcwd(), cls, ori_name))
        mask = cv2.imread(os.path.join(os.getcwd(), "Distracted Driver Dataset", destinationfolder, "test", ori_name + ".png"))
        W, H, C = ori_img.shape
        mask = cv2.resize(mask, (H, W))
        mask[mask < 128] = 1
        mask[mask > 128] = 0
        ori_img = ori_img * (mask == 0)
        ori_img[mask == 1] = 255
        cv2.imwrite(os.path.join(os.getcwd(), "Distracted Driver Dataset", finalfolder, "test", ori_name + ".png"),
                    ori_img)
        X_test.append(os.path.join(os.getcwd(), "Distracted Driver Dataset", finalfolder, "test", ori_name + ".png"))
        Y_test.append(int(naming(cls)[1]))

for cls in TRAIN_CLS:
    for ori_name in os.listdir(cls):
        ori_img = cv2.imread(os.path.join(os.getcwd(), cls, ori_name))
        mask = cv2.imread(os.path.join(os.getcwd(), "Distracted Driver Dataset", destinationfolder, "train", ori_name + ".png"))
        W, H, C = ori_img.shape
        mask = cv2.resize(mask, (H, W))
        mask[mask < 128] = 1
        mask[mask > 128] = 0
        ori_img = ori_img * (mask == 0)
        ori_img[mask == 1] = 255
        cv2.imwrite(os.path.join(os.getcwd(), "Distracted Driver Dataset", finalfolder, "train", ori_name + ".png"),
                    ori_img)
        X_test.append(os.path.join(os.getcwd(), "Distracted Driver Dataset", finalfolder, "train", ori_name + ".png"))
        Y_test.append(int(naming(cls)[1]))

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

(X_val, X_test_final, Y_val, Y_test_final) = train_test_split(X_test, Y_test, test_size=0.6, stratify=Y_test, random_state=42)
X_val = np.asarray(X_val)
Y_val = np.asarray(Y_val)
X_test_final = np.asarray(X_test_final)
Y_test_final = np.asarray(Y_test_final)
print(X_val.shape)
print(Y_val.shape)

X_train_resized = []
Y_train_resized = []
i = 0
for x in X_train:
    if len(x)!=0 and len(x[0])!=0:
        img = cv2.imread(x)
        img = cv2.resize(img, (75, 100))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img_blur = cv2.GaussianBlur(img, (3,3), 0)
        # edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
        # edges = np.asarray(edges)
        # edges = edges.reshape(120, 80, 1)
        X_train_resized.append(img/255.0)
        Y_train_resized.append(Y_train[i])
    i+=1
X_train_resized = np.asarray(X_train_resized)
X_train_resized = X_train_resized.reshape(-1, 100, 75, 1)
print("X_train_resized shape: ", X_train_resized.shape)

n, bins, patches = plt.hist(x=Y_train_resized, bins='auto', color='#0504aa')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Labels')
plt.ylabel('Frequency')

counts = []
for i in range(0, 8):
    counts.append(Y_train_resized.count(i))
np.argmin(counts)

df_resized_train = pd.DataFrame({"img": range(5455), "class": Y_train_resized})
print(df_resized_train)

from sklearn.utils import resample
X_train_sampled = []
Y_train_sampled = []
for c in range(0, 8):
    to_sample = df_resized_train[df_resized_train['class'] == c]
    downsample = resample(to_sample,
             replace=True,
             n_samples=counts[5],
             random_state=42)
    for index, val in downsample['img'].items():
        X_train_sampled.append(X_train_resized[val])
        Y_train_sampled.append(c)
X_train_sampled = np.asarray(X_train_sampled)
print(X_train_sampled.shape)

Y_train_sampled = np_utils.to_categorical(Y_train_sampled, 8)
print('New y_train shape: ', Y_train_sampled.shape)

X_val_resized = []
Y_val_resized = []
i = 0
for x in X_val:
    if len(x)!=0 and len(x[0])!=0:
        img = cv2.imread(x)
        img = cv2.resize(img, (75, 100))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img_blur = cv2.GaussianBlur(img, (3,3), 0)
        # edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
        # edges = np.asarray(edges)
        # edges = edges.reshape(120, 80, 1)
        X_val_resized.append(img/255)
        Y_val_resized.append(Y_train[i])
    i+=1
X_val_resized = np.asarray(X_val_resized)
X_val_resized = X_val_resized.reshape(-1, 100, 75, 1)
print(X_val_resized.shape)

Y_val_resized = np_utils.to_categorical(Y_val_resized, 8)
print('New y_train shape: ', Y_val_resized.shape)

model = models.Sequential()
model.add(layers.Conv2D(16, (2, 2), activation='relu', input_shape=(100, 75, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(32, (2, 2), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))

model.summary()

callbacks_list = [
    tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=2)
]

opt = tf.keras.optimizers.SGD(learning_rate=0.01)

model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 50
EPOCHS = 30

history = model.fit(X_train_sampled, Y_train_sampled, epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=callbacks_list,
                    validation_data=(X_val_resized, Y_val_resized), shuffle = True)

plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], "g--", label="Accuracy of training data")
plt.plot(history.history['val_accuracy'], "g", label="Accuracy of validation data")
plt.plot(history.history['loss'], "r--", label="Loss of training data")
plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

X_test_resized = []
Y_test_resized = []
i = 0
for x in X_test_final:
    if len(x)!=0 and len(x[0])!=0:
        img = cv2.resize(x, (75, 100))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        X_test_resized.append(img/255)
        Y_test_resized.append(Y_test_final[i])
    i+=1
X_test_resized = np.asarray(X_test_resized)
X_test_resized = X_test_resized.reshape(-1, 100, 75, 1)
print(X_test_resized.shape)

Y_test_resized = np_utils.to_categorical(Y_test_resized, 8)
print('New y_train shape: ', Y_test_resized.shape)

score = model.evaluate(X_test_resized, Y_test_resized, verbose=1)

print("\nAccuracy on test data: %0.2f" % score[1])
print("\nLoss on test data: %0.2f" % score[0])

LABELS = [0, 1, 2, 3, 4, 5, 6, 7]
def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

y_pred_test = model.predict(X_test_resized)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(Y_test_resized, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

print("\n--- Classification report for test data ---\n")

print(classification_report(max_y_test, max_y_pred_test))





