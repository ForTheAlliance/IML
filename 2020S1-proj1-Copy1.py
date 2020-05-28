
# coding: utf-8

# # The University of Melbourne, School of Computing and Information Systems
# # COMP90049 Introduction to Machine Learning, 2020 Semester 1
# -----
# ## Project 1: Understanding Student Success with Naive Bayes
# -----
# ###### Student Name(s): LIze Qin
# ###### Python version: Python 3
# ###### Submission deadline: 11am, Wed 22 Apr 2019

# This iPython notebook is a template which you will use for your Project 1 submission. 
# 
# Marking will be applied on the five functions that are defined in this notebook, and to your responses to the questions at the end of this notebook.
# 
# You may change the prototypes of these functions, and you may write other functions, according to your requirements. We would appreciate it if the required functions were prominent/easy to find. 

# In[31]:


from collections import defaultdict
import numpy as np
import pandas as pd

# likehood probability P(X|Y=ck)
likelihood = defaultdict(defaultdict) 
# Record the probabilities corresponding to all the values of this label
label_likelihood = defaultdict(defaultdict)
# the number of possible values for each feature Sj
Sj = defaultdict(float)


# In[32]:


# This function should open a data file in csv, and transform it into a usable format 
def load_data():    
    
    Student = pd.read_csv('student.csv',header = 0)
    return Student


# In[33]:


# This function should split a data set into a training set and hold-out test set
def split_data():
    
    Student = load_data()
    Shape = Student.shape[0]
    TrainRows = int(Shape*0.8)
    
    TrainSet = Student[0:TrainRows]
    TestSet = Student[TrainRows:]
    return TrainSet, TestSet

TrainSet, TestSet = split_data()


# In[34]:


# This function should build a supervised NB model   
def train():
    
    TrainSet, TestSet = split_data()

    # The first 29 attribute sets of the training set
    X = TrainSet.drop('Grade',axis=1,inplace=False) 
    y = TrainSet.columns
    SampleNum, FeatureNum = X.shape

    # grade
    grade = TrainSet.loc[:,'Grade']    
    GradeProb = defaultdict(float)
    GradeCounter = defaultdict(float)
    # number of samples   number of features


    # Calculate the possible values of grade and the number of it
    GradeLevel, Num_Gl = np.unique(grade, return_counts=True) 
    GradeCounter = dict(zip(GradeLevel, Num_Gl))

    for label, num_label in GradeCounter.items():
        # Calculate the prior probabilities
        GradeProb[label] = num_label / SampleNum 

    Attribute = y[:len(y)-1]

    # likehood probability P(X|Y=ck)
    likelihood = defaultdict(defaultdict) 
    # Record the probabilities corresponding to all the values of this label
    label_likelihood = defaultdict(defaultdict)
    # the number of possible values for each feature Sj
    Sj = defaultdict(float)
        
    for label in GradeLevel:
        #Get the number of other columns when they have same "grade"  A+ A B C D F
        GradeGroup = defaultdict(float)
        GradeGroup = TrainSet.loc[TrainSet['Grade']==label]  #The samples all have Grade'label'
        #print(GradeGroup)
        for i in Attribute:
            # Record the probability corresponding to each value of this label
            feature_val_prob = defaultdict(float)
            # Get the possible values of the features of the column and the number of occurrences of each value        
            feature_val, feature_cnt = np.unique(GradeGroup[i], return_counts=True)            
            #  Sj [i] records the number of possible values of the i-th column
            Sj[i] = feature_val.shape[0]            
            feature_counter = dict(zip(feature_val, feature_cnt))            
            for fea_val, cnt in feature_counter.items():                
                # Calculate the probability of each value of the list of features, and do Laplace smoothing λ= 1
                feature_val_prob[fea_val] = (cnt + 1) / (GradeCounter[label] + Sj[i])
                #print(feature_val_prob[fea_val])
            label_likelihood[i] = feature_val_prob        
        likelihood[label] = label_likelihood
        
        
    return likelihood, GradeProb, GradeCounter       


# In[57]:


# This function should predict the class for an instance or a set of instances, based on a trained model 

def predict(x):
    #Input samples and output their labels, essentially calculating the posterior probability
    #**To prevent underflow of floating point numbers,Logarithm turns multiplication function into summation function
    
            
    TrainSet, TestSet = split_data()
    column = TrainSet.columns
    likelihood, GradeProb, GradeCounter = train()
    # Save posterior probabilities classified into each label
    post_prob = defaultdict(float)
    # iterate each label to calculate the posterior probability
    for label, label_likelihood in likelihood.items():
        # get the dictionary of likelihood with same grade 'label'
        #prob = np.log(GradeProb[label])        
        prob = GradeProb[label]
        #print (label, label_likelihood)
        # Iterate each feature of the sample
        for i, fea_val in enumerate(x):
            feature_val_prob = label_likelihood[column[i]]
            #print (feature_val_prob)
            # If the feature value appears in the training set, then directly obtain the probability
            if fea_val in feature_val_prob:
                #print (feature_val_prob[fea_val])
                prob *= feature_val_prob[fea_val]
            else:
                # If the feature doesn't appear in the training set, use Laplacian smoothing to calculate the probability
                laplace_prob = 1 / (GradeCounter[label] + Sj[i])
                prob *= laplace_prob
        post_prob[label] = prob
        #print (post_prob[label])
    
    prob_list = list(post_prob.items())
    prob_list.sort(key=lambda x: x[1], reverse=True)
    #print (prob_list[0][0])
    
    return prob_list[0][0] # maximum


# In[58]:


# This function should evaluate a set of predictions in terms of accuracy
   
def evaluate():    
    TrainSet, TestSet = split_data()        
    TestGrade = TestSet["Grade"] 
       
    TestSet2 = TestSet.drop(columns="Grade")
    n_test = len(TestGrade)
    right = 0
    
      
    for i in range(0,n_test):
        pre_grade = predict(TestSet2.iloc[i])
        if pre_grade == TestGrade.iloc[i]:
            right += 1
        print (pre_grade,TestGrade.iloc[i])
    return right/n_test
    
a = evaluate()
a


# ## Questions (you may respond in a cell or cells below):
# 
# You should respond to Question 1 and two additional questions of your choice. A response to a question should take about 100–250 words, and make reference to the data wherever possible.
# 
# ### Question 1: Naive Bayes Concepts and Implementation
# 
# - a Explain the ‘naive’ assumption underlying Naive Bayes. (1) Why is it necessary? (2) Why can it be problematic? Link your discussion to the features of the students data set. [no programming required]
# - b Implement the required functions to load the student dataset, and estimate a Naive Bayes model. Evaluate the resulting classifier using the hold-out strategy, and measure its performance using accuracy.
# - c What accuracy does your classifier achieve? Manually inspect a few instances for which your classifier made correct predictions, and some for which it predicted incorrectly, and discuss any patterns you can find.
# 
# ### Question 2: A Closer Look at Evaluation
# 
# - a You learnt in the lectures that precision, recall and f-1 measure can provide a more holistic and realistic picture of the classifier performance. (i) Explain the intuition behind accuracy, precision, recall, and F1-measure, (ii) contrast their utility, and (iii) discuss the difference between micro and macro averaging in the context of the data set. [no programming required]
# - b Compute precision, recall and f-1 measure of your model’s predictions on the test data set (1) separately for each class, and (2) as a single number using macro-averaging. Compare the results against your accuracy scores from Question 1. In the context of the student dataset, and your response to question 2a analyze the additional knowledge you gained about your classifier performance.
# 
# ### Question 3: Training Strategies 
# 
# There are other evaluation strategies, which tend to be preferred over the hold-out strategy you implemented in Question 1.
# - a Select one such strategy, (i) describe how it works, and (ii) explain why it is preferable over hold-out evaluation. [no programming required]
# - b Implement your chosen strategy from Question 3a, and report the accuracy score(s) of your classifier under this strategy. Compare your outcomes against your accuracy score in Question 1, and explain your observations in the context of your response to question 3a.
# 
# ### Question 4: Model Comparison
# 
# In order to understand whether a machine learning model is performing satisfactorily we typically compare its performance against alternative models. 
# - a Choose one (simple) comparison model, explain (i) the workings of your chosen model, and (ii) why you chose this particular model. 
# - b Implement your model of choice. How does the performance of the Naive Bayes classifier compare against your additional model? Explain your observations.
# 
# ### Question 5: Bias and Fairness in Student Success Prediction
# 
# As machine learning practitioners, we should be aware of possible ethical considerations around the
# applications we develop. The classifier you developed in this assignment could for example be used
# to classify college applicants into admitted vs not-admitted – depending on their predicted
# grade.
# - a Discuss ethical problems which might arise in this application and lead to unfair treatment of the applicants. Link your discussion to the set of features provided in the students data set. [no programming required]
# - b Select ethically problematic features from the data set and remove them from the data set. Use your own judgment (there is no right or wrong), and document your decisions. Train your Naive Bayes classifier on the resulting data set containing only ‘unproblematic’ features. How does the performance change in comparison to the full classifier?
# - c The approach to fairness we have adopted is called “fairness through unawareness” – we simply deleted any questionable features from our data. Removing all problematic features does not guarantee a fair classifier. Can you think of reasons why removing problematic features is not enough? [no programming required]
# 

# In[ ]:


"""
Question 1  Answer:

a)	The ‘naïve’ assumption underlying Naïve Bayes :
    _ Features of an instance are conditionally independent given the class
    _ Instances are independent of each other
    _ The distribution of data in the training instances is the same as the distribution of data in the test instances
(1)	Necessary: With this assumption, the model works well.
(2)	Problematic: This assumption is not consistent with the facts in most cases, and these features are actually interrelated. 
    For example, learning time has a relationship with the grades.
b)	Before building a Naïve Bayes model, we need to calculate the prior probabilities and conditional probabilities. 
    For this dataset, the class which we want to predict is Final Grade (A+, A, B, C, D, F). So we need the prior probability of each label,
    and then calculate the the prior probabilities of each leavel of grade. 
    Next, recording the likelihood probabilities corresponding to each value of each grade level, complete the naive bayes model. 
    In prediction, we need to calculate the maximum of posterior probability based on each value of the input sample, and that is the prediction grade.
    (the code above for the implementation process,using laplace smoothing strategy)
c)  The accuracy of this classifier is 29%. Almost all the prediction grade is 'D',only one is 'A', there something wrong with this model.
    The problem might be in predict function. I need more time to deal with it. 

"""


# In[ ]:


"""
Question 2  Answer:

a) Precision = TP/(TP+FP)  it means what percentage of the compression we detected is correct
   Fecall = TP/(TP+FN)  it means for all accurate entries, how many were detected by us
   F-Measure is the weighted harmonic average of Precision and Recall:  Fβ = (1+β^2)PR/(Pβ^2+R)
   When parameter is 1, it is a F-1 measure.F1 combines the results of P and R. When F1 value is high, 
   it means that the experimental method is more reliable.
   Macro-averaging means calculating the statistical index value for each class, and then calculating 
   the arithmetic approximation for all classes.
   Micro-averaging means calculating the global confusion matrix for each instance in the data set 
   regardless of category, and then calculating the corresponding index.

b) For this model, we need to iterate 6 times, each time only a single score such as A is predicted, 
   and 6 different sets of TP / TN / FP / FN are obtained. Then use these 6 sets of data for calculation.


# In[ ]:


"""
Question 5  Answer:
a) The prediction may cause some illusions: for example, the student's performance is related to the student's gender, 
   which may cause the school to prefer students of a certain gender when enrolling students.（sex-grade in student dataset）
   
b) "sex,address,famisize,pstatus,reason,guardian"These redundant features have no substantial help in predicting academic performance, 
   and removing them may make the prediction results more accurate.
   
c) They have shown that simply avoiding the use of questionable features is insufficient for eliminating biases in determinations, 
   due to the indirect influence of questionable features. In addition, the influence of each feature on the results may have different weights,
   which is not considered by the simple classifier model. 

