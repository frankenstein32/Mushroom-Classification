# Mushroom-Classification
Classify different types of mushroom depending on their featues using SVM classifier.

__Dataset__ 
- Dataset contains the information regarding type of mushrooms along with their features like, Shape, Color, Size and ring type etc of the mushroom's cap.
- There are approx 23 features given for the each mushroom along with the type or can say label of the mushroom but the gata is given in the form of characters.<br> For e.g., 
![screenshot from 2018-12-13 21-54-22](https://user-images.githubusercontent.com/34310411/49952328-c0e46f80-ff21-11e8-90db-19d614d29cc3.png)

here, we need to first encode the data into numeric form and we can apply our algorithm. For encoding either one can use its own endcoding method by assiging each letter a particular number or can use Sklearn's LabelEncoder. About the predefined Label Encoder you can see the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html">documentation</a>.

__Algorithm__ : I have used Support Vector Machine to Classify the mushrooms in the given data. One can use Different algorithm inorder to predict the type of mushroom.

__Support vector Machine__ 
- Support Vector Machines are supervised learning models for classification and regression problems. They can solve linear and non-linear problems and work well for many practical problems.
- The idea of Support Vector Machines is simple: The algorithm creates a line which separates the classes in case e.g. in a classification problem.
- The goal of the line is to maximizing the margin between the points on either side of the so called decision line.
- The benefit of this process is, that after the separation, the model can easily guess the target classes (labels) for new cases.

__Essential Libraries__ : There are a few Libraries that one need to download inorder to run the given code.
- Pandas
- Matplotlib
- Numpy
- sklearn

__ScreenShots__

1. Command to run the code.


![screenshot from 2018-12-13 22-06-53](https://user-images.githubusercontent.com/34310411/49953763-d909be00-ff24-11e8-8d08-97fe3fb1b571.png)


2. Result of Score of mySVM and Sklearn's SVM.


![screenshot from 2018-12-13 22-21-06](https://user-images.githubusercontent.com/34310411/49954134-c643b900-ff25-11e8-8aa7-76065453b0dd.png)


3. Plot of the variation of loss Vs Epochs during the Learning of the Classifier.


![screenshot from 2018-12-13 22-14-19](https://user-images.githubusercontent.com/34310411/49953632-8c25e780-ff24-11e8-9d5e-83745a2443a6.png)
