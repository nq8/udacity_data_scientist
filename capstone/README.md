# Sparkify

## Motivation
In this project, the imaginary music streaming platform Sparkify is analyzed. The goal is to predict whether a user 
might quit his or her subscription. If we can predict accurately if a user is on the edge to cancel the subscription,
targeted special offers can be made. As the Sparkify company it is crucial to make these special offers only 
to the customers which would otherwise cancel. If Sparkify offers this treatment to the whole community, a significant 
amount of money is unnecessarily spent. 

## File Descriptions
- mini_sparkify_event_data.json: Small data set which is used to test my algorithm before the whole dataset is deployed 
executed on AWS
- jupyer file small: Jupyter notebook to process the small data set to test my algorithms

## Data Description
The full data set is 12GB and the mini data set is 123 MB.

## Approach
### Data Exploration and Analysis
This step is conducted to get a feeling how the data is structured and was the foundation of the decision which 
features to use.
### Feature Creation
The following features are created and used:  
1) thumbs down and up per UserId  
2) songs played  
3) duration used (max_ts - min_ts per UserId)  
4) songs per hour  
5) up/down per song ratio  
6) songs per hour  
7) errors per hour or day  
8) add to playlist per day
### Data Processing and Modelling
I used three different machine learning classification algorithms:
1) Logistic Regression
2) Random Forest
3) Gradient Boosted Tree
###
For evaluation purposes, the f1 score is used. This resulted in the following f1 scores:
- Linear Regression f1 score:
- Random Forest f1 score:
- Gradient Boosted Tree f1 score:

## Conclusion and potential Improvements

## References
This project is the final course project of the Data Scientist NanoDegree program offered by Udacity. Also the 
data set is provided by Udacity.
