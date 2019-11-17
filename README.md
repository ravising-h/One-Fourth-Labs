# One Fourth Labs
## INTERNSHIP PROGRAMMING ROUND 1


![]()
We are given a dataset containing 28x28 grayscale images. Each image is either a handwritten letter or digit. The dataset can be downloaded from here:
https://drive.google.com/file/d/12OYCKGQp1VybvLM157ioLU4Bjt7PWpt-/

Format of dataset:
The dataset is well-balanced and contains 47 classes, as described in the image below.
(10 digits, 26 capital letters and 11 small letters)



The dataset is present as a CSV file. You’ll find two CSV files: Train-set and Test-set.
You are supposed to train only using the train-set and use test-set only for calculating accuracy.

Number of samples:
Train set	: 112,800	(2400 images per class)
Test set	: 18,800	(  400 images per class)

##### CSV format:
Each line in the csv file corresponds to 1 sample. Each line will contain 785 values.
The first value in all lines indicate the label ID, and the remaining 784 values corresponds to the individual pixel values of the 28 X 28 image 
The *ASCII* value of each label ID can be found in the mapping.txt file.
For example, a label ID of 10 has an ASCII value of 65, which means that it corresponds to the character ‘A’.
You are supposed to use all the train samples (lines) to complete the following tasks:
Note: Each task must be submitted as independent runnable codes/Notebooks in a single GitHub repo. (i.e Don’t squeeze in all tasks in a single file/Model).
### Task 1: Letter/Digit Classifier (Easy)

Given an image, you must be able to classify whether the image is a letter or a digit.
**Expected outcome:**  You are expected to use a ML-based model (like CNNs, etc.) to solve the problem with a reasonably high accuracy.
### Task 2: Vowel/Consonant and Even/Odd Classifier (Moderate)
Given an image, you are supposed to design model(s) which does the following:
1. If the image is a letter, you are supposed to predict if it is a vowel or consonant.
2. If the image is a digit, you are supposed to predict if it is an even or odd number.

You are supposed to use only ML models that directly predicts the above, instead of doing manual predictions like using modulus operator on top of digit predictions.

**Expected outcome:**  Given an image, your end-to-end setup must print whether it is a letter or digit, and based on that, it must automatically run the corresponding model to print if it is vowel/consonant or even/odd respectively.

### Task 3: Character Classifier
Given an image, you are supposed to predict what digit or letter the image contains.
That is, you will be doing a classification task for 47 classes.

**Expected outcome:**  Given an image, you have to print what character it is (just using a single model). Also, report the class-wise accuracy if possible.


### My Method
I trained a basic model which classifies all 47 charecters. Then I used this pretrained model in Task 1 and Task 2. In Task 1 and 2, I used Decision Tree to classify Number, Letter and Odd, Even, Vowel, Const. In Task 3, I used 10 CNNs to classify charecters accurately.

**Procedure in Task 1 and 2:-**

1. Loaded data and visualized it.
2. Changed the input array from 784 to 28 X 28 X 1 and then divided by 255 to normalized it.
3. build the  CNN model and to classify the charecters.
4. Ran a Decision Tree model on the output of CNN.
5. Visualized and saved the output.

**Procedure of Task 3:-**

1. Loaded data and visualized it.
2. Changed the input array from 784 to 28 X 28 X 1 and then divided by 255 to normalized it.
3. build 10  CNN models and to classify the charecters.
4. Combine their results.
5. Visualized and saved the output.

### Thing I have Tried

1. Normalization
2. CNN
3. ANN
4. XGBoost
5. Weights Regularizers
6. Batch Normalization
7. Data Augmentation
8. Pruning
9. Stacked models
10. RandomCV

### Thing Worked for me

1. Normalization
2. CNN
3. Batch Normalization
4. Data Augmentation
5. Pruning
6. Stacked models



Models are stored in model folder.

#### utils.py has following functions:-
1. dataset_distribution - Dividing dataset into train validate test
2. one_hot_encoding - One hot encoding of Labels
3. de_encoding - Decoding of Labels
4. change_to_image - Change matrix from (784,) to (28,28,1)
5. create_download_link - creating downloading link
6. acc - printing accuracy, classification report
7. labelToDigitLetters - changing labels
8. labelToOddeven_Vowelcharecter - changing labels
9. It returns the CNN Model

To get prediction Run the jupyter file and in `predict` function give path to test_csv.


