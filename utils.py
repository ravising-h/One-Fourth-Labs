import numpy as np
from math import floor # Round OFF
import random 
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
def dataset_distribution(train,distribution = [60,20,20]):
    # dividing dataset in TRAIN, DEV, TEST
    # distribution is an array which tell divide percentage
    train = np.array(train)
    np.random.shuffle(train)
    perc_train = floor(distribution[0] * 0.01*train.shape[0])
    perc_dev = perc_train + floor(distribution[1] * 0.01*train.shape[0])
    perc_test = perc_dev + floor(distribution[2] * 0.01*train.shape[0])
    train_feature = train[0:perc_train,1:]
    train_label =  train[0:perc_train,0]
    
    dev_feature = train[perc_train:perc_dev,1:]
    dev_label =  train[perc_train:perc_dev,0]
    
    test_feature = train[perc_dev:perc_test,1:]
    test_label =  train[perc_dev:perc_test,0]
    
    return train_feature/255, train_label, dev_feature/255, dev_label, test_feature/255, test_label

def one_hot_encoding(label):
    no_of_class = np.unique(label).shape[0]
    enc_labels = np.zeros((label.shape[0],no_of_class))
    for index in range(label.shape[0]):
        enc_labels[index,label[index]] = 1
    return enc_labels


def de_encoding(prediction):
    predict = np.zeros((prediction.shape[0]))
    for index in range(prediction.shape[0]):
        predict[index] = np.argmax(prediction[index])
    return predict

def change_to_image(in_feature):
    feature = np.zeros(shape = (in_feature.shape[0],size_of_img[0],size_of_img[1], 1))
    for image_index in range(in_feature.shape[0]):
        feature[image_index] = np.fliplr(np.rot90(in_feature[image_index].reshape(size_of_img[0],size_of_img[1], 1),3))
    return feature


def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

def acc(y_test,prediction,cmap):
    cm = confusion_matrix(y_test, prediction)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    
    print ('Recall:', recall)
    print ('Precision:', precision)
    print ('\n clasification report:\n', classification_report(y_test,prediction))
    print ('\n confussion matrix:\n',confusion_matrix(y_test, prediction))
    plt.figure(figsize=(12, 13))
    ax = sns.heatmap(confusion_matrix(y_test, prediction),linewidths= 0.6,cmap = cmap)


def labelToDigitLetters(Labels):
  return np.where(Labels < 10 , 0, 1) ## Digits is 0 and Letters are 1

def labelToOddeven_Vowelcharecter(labels):
    ones = [0,2,4,6,8,10,14,18,24,30,36,40,44]
    return np.where(ones in labels,1,0 )