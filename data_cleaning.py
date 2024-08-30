import pandas as pd

df = pd.read_csv("D:\\Data_Science_Salary\\DataScience_Salary\\glassdoor_jobs.csv")

# salary parsing

df["Hourly"] = df["Salary Estimate"].apply(lambda x: 1 if "per hour" in x.lower() else 0)
df["Employer Provided"] = df["Salary Estimate"].apply(lambda x: 1 if "employer provided salary" in x.lower() else 0)
df = df[df["Salary Estimate"]!='-1']
salary = df["Salary Estimate"].apply(lambda x: x.split('(')[0])
rm_kd = salary.apply(lambda x: x.replace('K','').replace('$',''))
rm_ph = rm_kd.apply(lambda x:x.lower().replace('per hour','').replace("employer provided salary:",''))
df["Min Salary"] = rm_ph.apply(lambda x: int(x.split('-')[0]))
df["Max Salary"] = rm_ph.apply(lambda x: int(x.split('-')[1]))
df["Avg Salary"] = (df["Min Salary"] + df["Max Salary"]) / 2

# compant name text only

df["Company txt"] = df.apply(lambda x: x["Company Name"] if x["Rating"] < 0 else x["Company Name"][:-3],axis=1)

# state field

df["Job State"] = df["Location"].apply(lambda x: x.split(',')[1])
df["Job State"].value_counts()
df["Same State"] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0,axis=1)

# age of the company

df["Founded"] = df["Founded"].astype(int)
df["Age"] = df.Founded.apply(lambda x: -1 if x < 1 else 2024 - x)

# parsing of job discription (python, etc)
# python 
df["Python_yn"] = df["Job Description"].apply(lambda x: 1 if 'python' in x.lower() else 0)

# R-Studio
df["R_yn"] = df["Job Description"].apply(lambda x: 1 if 'r-studio' in x.lower() else 0)

# spark 
df["Spark_yn"] = df["Job Description"].apply(lambda x: 1 if 'spark' in x.lower() else 0)

# aws
df["Aws_yn"] = df["Job Description"].apply(lambda x: 1 if 'aws' in x.lower() else 0)

# excel
df["Excel_yn"] = df["Job Description"].apply(lambda x: 1 if 'excel' in x.lower() else 0)
df.Excel_yn.value_counts()

# Dropping redundant columns
df.drop("Unnamed: 0",axis=1,inplace=True)

df.to_csv("salary_cleaned2.csv",index=False)
