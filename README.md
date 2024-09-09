# Data Science Salary Analysis
- Created a tool that estimates data science salaries to help data scientists negotiate their income when they get a job.
- Scraped over 1000 job descriptions from glassdoor using python and selenium
- Engineered features from the text of each job description to quantify the value companies put on python, excel, aws, and spark.
- Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model.
- Built a client facing API using flask

## Resources  
**Python Version**: 3.10
**Scraper Github**: https://github.com/arapfaik/scraping-glassdoor-selenium
**For Web Framework Requirements**: pip install -r requirements.txt

## Data Cleaning
After scraping the data, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:

- Parsed numeric data out of salary
- Made columns for employer provided salary and hourly wages
- Removed rows without salary
- Parsed rating out of company text
- Made a new column for company state
- Added a column for if the job was at the company’s headquarters
- Transformed founded date into age of company
- Made columns for if different skills were listed in the job description:
  - Python
  - R
  - Excel
  - AWS
  - Spark
- Column for simplified job title and Seniority
- Column for description length

## EDA  
Analyzed the categorical and numerical data in order to understand what is the correlation between various features.
