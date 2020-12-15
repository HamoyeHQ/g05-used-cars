 
# Kaggle Project Description
## Context

Craigslist is the world's largest collection of used vehicles for sale, yet it's very difficult to collect all of them in the same place. I built a scraper for a school project and expanded upon it later to create this dataset which includes every used vehicle entry within the United States on Craigslist.

# Content

This data is scraped every few months, it contains most all relevant information that Craigslist provides on car sales including columns like <b> price, condition, manufacturer, latitude/longitude, and 18 other categories.</b> For ML projects, consider feature engineering on location columns such as long/lat. For previous listings, check older versions of the dataset.

# Problem Statement

* Craiglist being world leading company dealing on used cars, there is massive information about used cars in craiglist website and databases. The major challenge of any Analyst trying to handle craiglist data is that it is spread across various locations in US. So, how can we analyze this dataset to get useful business insights about used cars in USA? Can a machine learning model to built using this dataset? These are few of many questions that we want to find answers to in this project.

# Project Objectives
* To understand the underlying patterns in various features of the dataset.
* To generate meaningful business insights from the dataset.
* To generate a preprocessed dataset that can be used for ML modelling.
* To build a ML model that can predict the price of used cars in USA.
* To build a ML workflow pipeline for this project.
* To build a web interactive interface for end user inferences.


# Introduction
In many countries of the world, it is common to shop for used cars rather than buy new ones, this is a decision that is mostly informed by the financial cost of new vehicles or sometimes the belief that used cars work better or last longer than new ones. However, there is always ambiguity in the price of the car in question, as the resale price differs from seller to seller, with some expecting or requiring long negotiation before the final sale. Apart from these subjective factors, there are physical features that play a part in the price of a used car; such as the age, manufacturer, model, and mileage.
This project looks at predicting the price given for used cars based on these features.

# Dataset
The dataset used in this project comes from Kaggle and it contains scraped data from Craigslist, an American classified advertisements website, with one of the world’s largest collection of used vehicles for sale. The dataset had information on 458,213 cars with 25 columns of the features of the car, the location it is being sold from, and when it was advertised for sale.
As the information in the dataset was user inputted, there were a large number of missing values and error inputs, requiring a lot of cleaning

A significant number of the columns in the dataset had a large percentage of missing data, our cleaning process began with an in-depth exploration of the dataset features, which led to the discovery that the description column contained the missing information on the condition of the car, as well as some other physical features, in an unstructured format. This called for a meticulous extraction of data as we cleaned and processed the data, filling missing values with information from the description column when available and dropping rows and columns when needed. Outliers and error inputs were also removed using quartiles. At the end of this process we ended with a clean dataset of 186923 rows out of the initial 458213 entries.
Our next step was to perform some analysis on the dataset, exploring the dataset features and seeing how they correspond against price.
The ascendance of some particular categories of used cars is worth mentioning, with the diesel, red and 4 wheel-drive options outselling their counterparts on average.

# Modelling
In order to predict the price of a used car, there were a few features we picked up as important to the price variable (region, year, manufacturer, model, cylinders, fuel, odometer, title_status, transmission, drive, type, paint_color, state).
As there were a number of categorical variables in our dataset, a label encoder was first used while the numerical variables were standardized and scaled. This standardized dataset was tested with several models like Linear Regression, Decision Trees, Random Forest, Extra Trees Regressor, Catboost, LGBM, XGBRF Regressor.
From all the models used, Extra Trees Regressor gave the highest R2 score of 0.928 with the lowest mae value of 1489.648

[Link to project medium article](https://medium.com/@sootersaalu/open-source-documentation-predictive-analysis-on-price-of-used-cars-e435397ea133)

[Link to project presentation slides](https://docs.google.com/presentation/d/1wozqkrS3h3GUjS1DYf4HoLDlA-x1RhLuU_OsGv7xjdQ/edit?usp=sharing)
