# AirBnB Data Analysis for Seattle

## Motivation
In this project, I will examine how the price for apartments in Seattle correlate to different features such as the number of bed rooms, the apartment type or the rating score.

## Question 1: Do better review scores correlate with higher prices?
## Answer 1: A small positive correlation between the price and the review score rating is found.

## Question 2: How is the distribution of apartment sizes in Seattle at airBnB?
## Answer 2: The mean apartment size is 854 square feet with a minimal size of 420 and maximal size of 3000.

## Question 3: What kind of property type has the highest price?
## Answer 3: The maximum price per night is made using a boat while the apartment type with the lowest price is a dorm.

## Question 4: Let's create a linear regression model for the price using the apartment size, property type, number of beds and review score to predict the price per night. What will be the most influencing factor?
## Answer 4: We can see, that the strongest correlation between the price and the input features is the number of beds. Also, the apartment type does not matter that much anymore. But we should take this insight with a grain of salt, since the number of input features is small compared to the original data set.


## File Descriptions

```
./files/seattle_listing.csv
```
This file contains a detailed data base for numerous available apartments in Seattle. The data can be found under [https://www.kaggle.com/airbnb/seattle/data].

## Used libraries
Besides popular and standard libraries such as sklearn and numpy+pandas, no additional libraries are used.