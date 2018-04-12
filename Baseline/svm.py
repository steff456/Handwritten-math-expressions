import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import svm
from charge import save_var

train = np.load('HOG_train.npy')
test = np.load('HOG_test.npy')
train = train.item()
test = test.item()

train_data = []
train_labels = []
for key in train.keys():
    act_hog = train[key]
    for i in range(0, len(act_hog)):
        train_labels.append(key)
        train_data.append(act_hog[i])

test_data = []
test_labels = []
for key in train.keys():
    act_hog = test[key]
    for i in range(0, len(act_hog)):
        test_labels.append(key)
        test_data.append(act_hog[i])

lin_clf = svm.LinearSVC()
lin_clf.fit(train_data, train_labels)
test_pred = []
for i in range(0, len(test_data)):
    pred_act = lin_clf.predict(test_data[i].reshape(1, -1))
    test_pred.append(pred_act[0])

conf = confusion_matrix(test_labels, test_pred)
class_names = list(train.keys())
norm = conf / conf.astype(np.float).sum(axis=1, keepdims=True)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = '.' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# save_var('conf_m.npy', conf)
ACA = sum(np.diagonal(norm))
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
