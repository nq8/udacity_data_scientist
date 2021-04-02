# Sparkify

## Motivation
In this project, the imaginary music streaming platform Sparkify is analyzed. The goal is to predict whether a user 
might quit his or her subscription. If we can predict accurately if a user is on the edge to cancel the subscription,
targeted special offers can be made. As the Sparkify company it is crucial to make these special offers only 
to the customers which would otherwise cancel. If Sparkify offers this treatment to the whole community, a significant 
amount of money is unnecessarily spent. 

## File Descriptions
- Sparkify_big_data.ipynb: Jupyter notebook to process the big data set to test my algorithms
- Sparkify_small_data.ipynb: Jupyter notebook to process the small data set to test my algorithms
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
3) duration used Sparkify
4) up/down per song ratio  
5) errors per hour or day  
6) add to playlist per day
### Data Processing and Modelling
I used three different machine learning classification algorithms:
1) Logistic Regression
2) Random Forest
3) Gradient Boosted Tree
###
For evaluation purposes, the f1 score is used. This resulted in the following f1 scores:
- Linear Regression f1 score: 0.739
- Random Forest f1 score: 0.745
- Gradient Boosted Tree f1 score: 0.759

## Conclusion and potential Improvements
This machine learning model potentially saves Sparkify a significant amount of money. Details on this analysis 
can be found in my [blog post](https://nniillss.medium.com/should-i-stay-or-should-i-go-the-answer-is-stay-always-6bfebecde53b). 

## References
This project is the final course project of the Data Scientist NanoDegree program offered by Udacity. Also the 
data set is provided by Udacity.

Link to small data set: s3n://udacity-dsnd/sparkify/mini_sparkify_event_data.json
Link to big data set: s3n://udacity-dsnd/sparkify/sparkify_event_data.json
