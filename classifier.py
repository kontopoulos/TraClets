import os
import cv2
import argparse
import datetime
from imutils import paths
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.ensemble import RandomForestClassifier

# features description -1:  Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-description -2 Color Histogram
def fd_histogram(image, mask=None):
    # conver the image to HSV colors-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #COMPUTE THE COLOR HISTPGRAM
    hist  = cv2.calcHist([image],[0,1,2],None,[11,11,11], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist,hist)
    # return the histogram
    return hist.flatten()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-f", "--folds", type=int, help="path and name to output model")
args = vars(ap.parse_args())

data = []
labels = []
times = []
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    start = datetime.datetime.now()
    fv_hu_moments = fd_hu_moments(image)
    fv_histogram  = fd_histogram(image)
    global_feature = np.hstack([fv_histogram, fv_hu_moments])
    end = datetime.datetime.now()
    delta = end - start
    ms = int(delta.total_seconds() * 1000)
    times.append(ms)
    data.append(global_feature)
    # extract the class label from the image path and update the labels list
    label = int(imagePath.split(os.path.sep)[-2])
    labels.append(label)

print(f'[INFO] Average feature extraction time: {sum(times)/len(times)} ms.')
data = np.array(data)
labels = np.array(labels)

precisions = []
recalls = []
f1_scores = []

num_folds = args["folds"]
current_fold = 1
kf = KFold(n_splits=num_folds, shuffle=True)
best_model = 0
model_score = 0.0
for train_index, test_index in kf.split(data):
    print(f'[INFO] Current fold: {current_fold}')
    trainX, testX = data[train_index], data[test_index]
    trainY, testY = labels[train_index], labels[test_index]
    clf = RandomForestClassifier(n_estimators=100,max_depth=15,random_state=15)
    start = datetime.datetime.now()
    clf.fit(trainX,trainY)
    end = datetime.datetime.now()
    delta = end - start
    ms = int(delta.total_seconds() * 1000)
    print(f'[INFO] Training time: {ms} ms.')
    start = datetime.datetime.now()
    predictions = clf.predict(testX)
    end = datetime.datetime.now()
    delta = end - start
    ms = int(delta.total_seconds() * 1000)
    print(f'[INFO] Average classification time: {ms/testX.size} ms | Total classification time: {ms} ms.')
    precision,recall,fscore,support=score(testY,predictions,average='macro')
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(fscore)
    if fscore > model_score:
        best_model = clf
        model_score = fscore
    print('[INFO] Precision : {}'.format(precision))
    print('[INFO] Recall    : {}'.format(recall))
    print('[INFO] F-score   : {}'.format(fscore))
    current_fold += 1

precisions = np.array(precisions)
recalls = np.array(recalls)
f1_scores = np.array(f1_scores)
print('========== VALIDATION RESULTS ==========')
print(f'[INFO] Average Precision : {np.mean(precisions)} | Standard Deviation: {np.std(precisions)}')
print(f'[INFO] Average Recall : {np.mean(recalls)} | Standard Deviation: {np.std(recalls)}')
print(f'[INFO] Average F1-score : {np.mean(f1_scores)} | Standard Deviation: {np.std(f1_scores)}')