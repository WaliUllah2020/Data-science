# Data-science
Data science is a difficult field. There are many reasons for this, but the most important one is that it requires a broad set of skills and knowledge. The core elements of data science are math, statistics, and computer science. The math side includes linear algebra, probability theory, and statistics theory
 # the importance of Data Science:
 Data Science enables enterprises to measure, track, and record performance metrics for facilitating enterprise-wide enhanced decision making. Companies can analyze trends to make critical decisions to engage customers better, enhance company performance, and increase profitability.
 # the process of Data Science:
 Data science is the process of extracting insights and knowledge from data using various techniques and tools. The general process of data science can be broken down into the following steps:
* Problem definition: The first step is to identify and define the problem or question that needs to be answered. This includes understanding the business objective and the relevant data sources.
* Data collection: The second step is to collect the necessary data. This can involve gathering data from existing databases, scraping data from websites, or collecting new data through surveys or experiments.
* Data cleaning and preparation: The third step involves cleaning and preparing the data for analysis. This includes removing duplicates, dealing with missing values, and transforming the data into a format suitable for analysis.
* Exploratory data analysis: In this step, the data is explored and visualized to gain a better understanding of the relationships and patterns within the data.
* Feature engineering: Feature engineering involves creating new features from the existing data to improve the performance of the model.
* Model selection and training: The next step is to select an appropriate model for the problem at hand and train it on the data.
* Model evaluation: The model is evaluated using various metrics to determine its performance and whether it is suitable for the problem at hand.
* Model deployment: Once the model has been evaluated and deemed suitable, it can be deployed in a production environment to generate predictions or recommendations.
* Monitoring and maintenance: Finally, the model must be monitored and maintained to ensure it continues to perform well and remains up-to-date with any changes in the data or business environment.
This process is iterative, meaning that it may involve going back and forth between different steps as new information or insights are discovered.
# Data Science Tools And Technologies: 
There are a wide range of tools and technologies used in data science, depending on the specific task or problem at hand. Some commonly used tools and technologies include:
* Programming languages: Data scientists often use programming languages such as Python, R, and SQL to manipulate and analyze data, as well as to build machine learning models.
* Data visualization tools: Visualization tools like Tableau, PowerBI, and matplotlib are used to create charts, graphs, and other visualizations that help to communicate insights and patterns in the data.
* Big data technologies: Big data technologies like Apache Hadoop, Apache Spark, and Apache Kafka are used to process, store, and analyze large volumes of data.
* Machine learning frameworks: Machine learning frameworks like TensorFlow, PyTorch, and scikit-learn are used to build and train machine learning models.
* Cloud computing platforms: Cloud platforms like AWS, Azure, and Google Cloud are used to store and process data, as well as to deploy machine learning models.
* Database management systems: Database management systems like MySQL, PostgreSQL, and MongoDB are used to store and manage data.
* Data integration tools: Data integration tools like Apache Nifi and Talend are used to extract, transform, and load data from various sources.
* Text analysis tools: Text analysis tools like NLTK, spaCy, and Gensim are used to analyze and extract insights from text data
* Statistical analysis software: Statistical analysis software like SAS, SPSS, and STATA are used to analyze data using statistical methods.
These are just a few examples of the many tools and technologies used in data science. The choice of tools and technologies depends on the specific task or problem at hand, as well as the preferences and expertise of the data scientist or data science team.
# Data Science Skill:
Data science requires a combination of technical and non-technical skills. Some of the key skills required for data science include:
* Programming skills: Data scientists must have strong programming skills, particularly in languages such as Python and R. They should also be familiar with database query languages like SQL.
* Statistics and mathematics: Data scientists should have a solid understanding of statistical and mathematical concepts, including probability, linear algebra, and calculus.
* Data visualization: Data scientists must be able to effectively communicate insights from data through visualizations, so they need to have experience with data visualization tools like Tableau, PowerBI, or matplotlib.
* Machine learning: Data scientists should have a deep understanding of machine learning algorithms, including supervised and unsupervised learning, as well as experience with machine learning libraries like scikit-learn, TensorFlow, and PyTorch.
* Data cleaning and preprocessing: Data scientists should be skilled in cleaning and preprocessing data, including dealing with missing values, data normalization, and data transformation.
* Domain knowledge: Data scientists should have domain-specific knowledge in the industry or field they are working in. For example, a data scientist working in healthcare should have an understanding of healthcare data and related regulations.
* Problem-solving: Data scientists should have strong problem-solving skills to identify and define problems, and come up with creative solutions using data.
* Communication: Data scientists should have excellent communication skills to effectively convey complex data insights and findings to both technical and non-technical audiences.
* Collaboration: Data scientists should be able to collaborate effectively with other members of a team, including data engineers, data analysts, and business stakeholders.
These are just some of the key skills required for data science. The specific skills required may vary depending on the industry or application domain.
..........................................................................................................................................
# Python
# Numpy
# Series And Dataframe
# introduction to Python
Python is a popular and versatile programming language known for its simplicity, readability, and flexibility. Created by Guido van Rossum and first released in 1991, Python has since gained immense popularity and is widely used in various domains, including web development, data analysis, artificial intelligence, scientific computing, automation, and more.
# Features of Python:
 Readable and Elegant Syntax: Python emphasizes a clean and straightforward syntax that is easy to understand, making it an ideal language for beginners and experienced developers alik
Interpreted Language: Python is an interpreted language, meaning the code is executed line by line at runtime, rather than being compiled into machine code beforehand.
Multi-paradigm Language: Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming.
Large Standard Library: Python comes with a vast standard library that offers numerous pre-built modules and functions, making it convenient for developers to perform a wide range of tasks without having to write everything from scratch.
Open-source and Community-driven: Python is open-source, which means its source code is freely available for everyone. It has a large and active community of developers who contribute to its growth and development.
Cross-platform Compatibility: Python code can run on various operating systems, including Windows, macOS, Linux, and more.
# Code
import requests
import json
import pandas as pd
responce = requests.get('https://api.themoviedb.org/3/movie/top_rated?api_key=0f30a646e4c3e66f16369dc1f453fc90');
a = responce.json();
 puts data in DataFrame
df2 = pd.DataFrame.from_dict(a);
 data frame contained nested dictionary so we seperated the result columb as a dictionary
df= df2['results'].to_dict();
 I created an other Data frame for our results .
df1 = pd.DataFrame.from_dict(df);
print(df1)
 Series And Dataframe
In Python, the pandas library provides two essential data structures for data manipulation and analysis: Series and DataFrame.
# Series:
A Series is a one-dimensional labeled array that can hold data of any type (integer, string, float, etc.). It is similar to a one-column table or a Python list with labeled indices. The indices help to access and align data easily.
# Code
import pandas as pd
 Creating a Series from a Python list
data = [10, 20, 30, 40, 50]
series = pd.Series(data)
 Creating a Series with custom indices
series_custom_index = pd.Series(data, index=['A', 'B', 'C', 'D', 'E'])
# DataFrame:
A DataFrame is a two-dimensional labeled data structure, like a spreadsheet or SQL table. It consists of rows and columns and can hold multiple data types. Each column in a DataFrame is a Series.
# Code
import pandas as pd
 Creating a DataFrame from a dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'San Francisco', 'Los Angeles']
}
df = pd.DataFrame(data)
 Specifying custom row indices
df_custom_index = pd.DataFrame(data, index=['person1', 'person2', 'person3'])
#  introduction to Numpy
NumPy (Numerical Python) is a powerful Python library for numerical and scientific computing. It provides support for large, multi-dimensional arrays and matrices, along with an extensive collection of mathematical functions to operate on these arrays. NumPy is the foundation of many other Python libraries used in data analysis, machine learning, and scientific research.
# Key features of NumPy:
 Ndarray: The ndarray (n-dimensional array) is the fundamental data structure in NumPy. It is a homogenous array of fixed-size items that allows you to perform efficient operations on large datasets.
Vectorized Operations: NumPy enables vectorized operations, meaning you can apply mathematical operations to entire arrays without using explicit loops. This leads to faster and more efficient computations.
Broadcasting: NumPy allows broadcasting, which enables operations on arrays with different shapes, making it more convenient to work with data of varying dimensions.
Mathematical Functions: NumPy comes with a comprehensive set of mathematical functions, including basic arithmetic, statistical, trigonometric, and linear algebra operations.
Integration with C/C++ and Fortran: NumPy provides a C API, making it easy to integrate with low-level languages like C, C++, and Fortran, which enhances performance in numerical computations.
Random Number Generation: NumPy includes functions for generating random numbers with different distributions, crucial in simulations and statistical analysis.
#Getting Started with NumPy:
To use NumPy, you need to install it on your system. You can install it using pip:
# Code
 You'll recall that we import a library using the `import` keyword as numpy's common abbreviation is np
import numpy as np
import math
 Arrays are displayed as a list or list of lists and can be created through list as well. When creating an
 array, we pass in a list as an argument in numpy array
a = np.array([1, 2, 3])
a = np.array([[1,2,3],[1,2,3]])
print(a)
 We can print the number of dimensions of a list using the ndim attribute
print(a.ndim)
 If we pass in a list of lists in numpy array, we create a multi-dimensional array, for instance, a matrix
b = np.array([[1,2,3],[4,5,6]])
b
array([[1, 2, 3],
       [4, 5, 6]])
 We can print out the length of each dimension by calling the shape attribute, which returns a tuple
b.shape
(2, 3)
.........................................................................................................................................
# Data Types and Sources
# types of Fetching And Data Through the API
When working with APIs (Application Programming Interfaces), there are generally two types of data fetching methods:
#HTTP Methods for Data Fetching:
The most common HTTP methods used for fetching data through APIs are:
GET: This method is used to retrieve data from the server. When you make a GET request to a specific API endpoint, the server returns the requested data.
POST: This method is used to send data to the server for processing. It is often used to create or update resources on the server.
PUT: Similar to POST, PUT is used to update existing resources on the server.
DELETE: As the name suggests, this method is used to delete a resource on the server.
These methods are part of the HTTP protocol and are commonly used to interact with RESTful APIs, which follow the principles of Representational State Transfer (REST).
#Data Formats for API Response:
APIs can return data in various formats. The most commonly used data formats for API responses are:
JSON (JavaScript Object Notation): JSON is a lightweight data interchange format that is easy for both humans and machines to read and write. It is widely used in web APIs for data exchange.
XML (eXtensible Markup Language): XML is another data format used in APIs. It provides a way to structure data using tags, similar to HTML. While less popular than JSON in modern APIs, XML is still used in some cases.
CSV (Comma-Separated Values): CSV is a simple tabular format where each row represents a data record, and each column represents a field. It is commonly used for tabular data exchange.
HTML (Hypertext Markup Language): In some cases, APIs return data in HTML format, especially for web scraping purposes
Binary Formats (e.g., images, audio, video): APIs can also be used to fetch binary data, such as images, audio, and video files.
When interacting with an API, you need to be aware of the data format used in the API response. You'll typically parse the response data (e.g., using JSON decoding) to extract and use the relevant information in your application.
# Code
import pandas as pd
import requests
response = requests.get('https://api.themoviedb.org/3/movie/top_rated?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US&page=1')
temp_df = pd.DataFrame(response.json()['results'])[['id','title','overview','release_date','popularity','vote_average','vote_count']]
df.head()
TMDB API : https://developers.themoviedb.org/ RapidAPI : https://rapidapi.com/collection/list-of-free-apis JSON Viewer: http://jsonviewer.stack.hu/
import pandas as pd
import seaborn as sns
df = pd.read_csv('train.csv')
.........................................................................................................................................
# Data Cleaning and Preprocessing
# Pivot Table Scale Meging Dataframe Groupby
# Data Cleaning and Preprocessing
Data cleaning and preprocessing are critical steps in the data analysis and machine learning pipeline. They involve transforming raw, noisy, or inconsistent data into a clean and structured format suitable for analysis and modeling. The goal is to improve data quality, remove errors, handle missing values, and ensure the data is in a usable state for further analysis or training machine learning models.
Here are some common data cleaning and preprocessing techniques:

Handling Missing Values:
Missing data is a common issue in datasets. It's essential to deal with missing values appropriately to avoid bias and errors in the analysis. Common strategies include:

Removing rows or columns with missing values (only when the missing data is negligible).
Imputing missing values using statistical measures like mean, median, or mode.
Using advanced imputation techniques like K-nearest neighbors or interpolation.
Data Transformation:
Data may need to be transformed to bring it to a consistent and comparable scale. Common transformations include:

Scaling: Standardizing data to have a mean of 0 and a standard deviation of 1.
Normalization: Scaling data to a specific range, often between 0 and 1.
Log Transformation: Taking the logarithm of skewed data to make it more normally distributed.
Handling Outliers:
Outliers are extreme values that differ significantly from the majority of the data. They can affect the performance of machine learning models. Options for handling outliers include:
Removing outliers (only when they are erroneous data points).
Capping or flooring outliers to a specific value.
Transforming the data to reduce the impact of outliers.
Encoding Categorical Variables:
Machine learning algorithms typically require numerical inputs, so categorical variables need to be encoded. Common encoding techniques include one-hot encoding and label encoding.
Feature Engineering:
Feature engineering involves creating new features or modifying existing ones to improve the predictive power of machine learning models. This may include extracting information from text, dates, or numerical data or creating interaction terms.
Data Standardization:
Standardizing data ensures that all features have a mean of 0 and a standard deviation of 1. This is particularly important for algorithms that rely on distance calculations, like K-nearest neighbors or SVM.
Handling Imbalanced Data:
Imbalanced data occurs when one class dominates the dataset, leading to biased model performance. Techniques like oversampling, undersampling, or using class weights can address this issue.
Data Integration:
In some cases, data comes from multiple sources and needs to be integrated or merged into a single dataset for analysis.
Data Reduction:
When dealing with large datasets, data reduction techniques like PCA (Principal Component Analysis) or feature selection can be used to reduce the dimensionality of the data while preserving important information.
### Pivot Table
A pivot table is a way of summarizing data in a DataFrame for a particular purpose. It makes heavy use of the aggregation function. A pivot table is itself a DataFrame, where the rows represent one variable that you're interested in, the columns another, and the cell's some aggregate value. A pivot table also tends to includes marginal values as well, which are the sums for each column and row. This allows you to be able to see the relationship between two variables at just a glance.
### Code
 Lets take a look at pivot tables in pandas
import pandas as pd
import numpy as np
 Here we have the Times Higher Education World University Ranking dataset, which is one of the most
 influential university measures. Let's import the dataset and see what it looks like
df = pd.read_csv('datasets/cwurData.csv')
df.head()
def create_category(ranking):
     Since the rank is just an integer, I'll just do a bunch of if/elif statements
    if (ranking >= 1) & (ranking <= 100):
        return "First Tier Top Unversity"
    elif (ranking >= 101) & (ranking <= 200):
        return "Second Tier Top Unversity"
    elif (ranking >= 201) & (ranking <= 300):
        return "Third Tier Top Unversity"
    return "Other Top Unversity"

 Now we can apply this to a single column of data to create a new series
df['Rank_Level'] = df['world_rank'].apply(lambda x: create_category(x))
 And lets look at the result
df.head()
df.pivot_table(values='score', index='country', columns='Rank_Level', aggfunc=[np.mean]).head()
df.pivot_table(values='score', index='country', columns='Rank_Level', aggfunc=[np.mean, np.max]).head()
df.pivot_table(values='score', index='country', columns='Rank_Level', aggfunc=[np.mean, np.max], margins=True).head()
new_df=df.pivot_table(values='score', index='country', columns='Rank_Level', aggfunc=[np.mean, np.max], 
               margins=True)
 Now let's look at the index
print(new_df.index)
 And let's look at the columns
print(new_df.columns)

### Scale Meging/Meging Dataframe 
In this lecture we're going to address how you can bring multiple dataframe objects together, either by merging them horizontally, or by concatenating them vertically. Before we jump into the code, we need to address a little relational theory and to get some language conventions down. I'm going to bring in an image to help explain some concepts.

Venn Diagram

Ok, this is a Venn Diagram. A Venn Diagram is traditionally used to show set membership. For example, the circle on the left is the population of students at a university. The circle on the right is the population of staff at a university. And the overlapping region in the middle are all of those students who are also staff. Maybe these students run tutorials for a course, or grade assignments, or engage in running research experiments.
So, this diagram shows two populations whom we might have data about, but there is overlap between those populations.
When it comes to translating this to pandas, we can think of the case where we might have these two populations as indices in separate DataFrames, maybe with the label of Person Name. When we want to join the DataFrames together, we have some choices to make. First what if we want a list of all the people regardless of whether they're staff or student, and all of the information we can get on them? In database terminology, this is called a full outer join. And in set theory, it's called a union. In the Venn diagram, it represents everyone in any circle.

Here's an image of what that would look like in the Venn diagram.
Union
It's quite possible though that we only want those people who we have maximum information for, those people who are both staff and students. Maybe being a staff member and a student involves getting a tuition waiver, and we want to calculate the cost of this. In database terminology, this is called an inner join. Or in set theory, the intersection. It is represented in the Venn diagram as the overlapping parts of each circle.
## Code
mport pandas as pd
 First we create two DataFrames, staff and students.
staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR'},
                         {'Name': 'Sally', 'Role': 'Course liasion'},
                         {'Name': 'James', 'Role': 'Grader'}])
 And lets index these staff by name
staff_df = staff_df.set_index('Name')
 Now we'll create a student dataframe
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},
                           {'Name': 'Mike', 'School': 'Law'},
                           {'Name': 'Sally', 'School': 'Engineering'}])
 And we'll index this by name too
student_df = student_df.set_index('Name')
 And lets just print out the dataframes
print(staff_df.head())
print(student_df.head())
pd.merge(staff_df, student_df, how='outer', left_index=True, right_index=True)
pd.merge(staff_df, student_df, how='left', left_index=True, right_index=True)
 First, lets remove our index from both of our dataframes
staff_df = staff_df.reset_index()
student_df = student_df.reset_index()
 student DataFrames that have a location information added to them.
staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR', 
                          'Location': 'State Street'},
                         {'Name': 'Sally', 'Role': 'Course liasion', 
                          'Location': 'Washington Avenue'},
                         {'Name': 'James', 'Role': 'Grader', 
                          'Location': 'Washington Avenue'}])
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business', 
                            'Location': '1024 Billiard Avenue'},
                           {'Name': 'Mike', 'School': 'Law', 
                            'Location': 'Fraternity House #22'},
                           {'Name': 'Sally', 'School': 'Engineering', 
                            'Location': '512 Wilson Crescent'}])
pd.merge(staff_df, student_df, how='left', on='Name')
Here's an example with some new student and staff data
staff_df = pd.DataFrame([{'First Name': 'Kelly', 'Last Name': 'Desjardins', 
                          'Role': 'Director of HR'},
                         {'First Name': 'Sally', 'Last Name': 'Brooks', 
                          'Role': 'Course liasion'},
                         {'First Name': 'James', 'Last Name': 'Wilde', 
                          'Role': 'Grader'}])
student_df = pd.DataFrame([{'First Name': 'James', 'Last Name': 'Hammond', 
                            'School': 'Business'},
                           {'First Name': 'Mike', 'Last Name': 'Smith', 
                            'School': 'Law'},
                           {'First Name': 'Sally', 'Last Name': 'Brooks', 
                            'School': 'Engineering'}])
%%capture
df_2011 = pd.read_csv("datasets/college_scorecard/MERGED2011_12_PP.csv", error_bad_lines=False)
df_2012 = pd.read_csv("datasets/college_scorecard/MERGED2012_13_PP.csv", error_bad_lines=False)
df_2013 = pd.read_csv("datasets/college_scorecard/MERGED2013_14_PP.csv", error_bad_lines=False)
 Let's get a view of one of the dataframes
df_2011.head(3)
 We see that there is a whopping number of columns - more than 1900! We can calculate the length of each
 dataframe as well
print(len(df_2011))
print(len(df_2012))
print(len(df_2013))

frames = [df_2011, df_2012, df_2013]
pd.concat(frames)

 Now let's try it out
pd.concat(frames, keys=['2011','2012','2013'])
### Scales
 Let's bring in pandas as normal
import pandas as pd
df=pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],
                index=['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 
                       'ok', 'ok', 'ok', 'poor', 'poor'],
               columns=["Grades"])
df
df.dtypes
df["Grades"].astype("category").head()
 We see now that there are eleven categori
my_categories=pd.CategoricalDtype(categories=['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+']                    ordered=True)
grades=df["Grades"].astype(my_categories)
grades.head()
df[df["Grades"]>"C"]
grades[grades>"C"]
### Groupby/Group Data
Sometimes we want to select data based on groups and understand aggregated data on a group level. We have seen that even though Pandas allows us to iterate over every row in a dataframe, it is generally very slow to do so. Fortunately Pandas has a groupby() function to speed up such task. The idea behind the groupby() function is that it takes some dataframe, splits it into chunks based on some key values, applies computation on those chunks, then combines the results back together into another dataframe. In pandas this is referred to as the split-apply-combine pattern.
### Code
import pandas as pd
import numpy as np
 Let's look at some US census data
df = pd.read_csv('datasets/census.csv')
 And exclude state level summarizations, which have sum level value of 40
df = df[df['SUMLEV']==50]
df.head()
 Let's run such task for 3 times and time it. For this we'll use the cell magic function %%timeit
%%timeit -n 3 #magic command 

for state in df['STNAME'].unique():
     We'll just calculate the average using numpy for this particular state
    avg = np.average(df.where(df['STNAME']==state).dropna()['CENSUS2010POP'])
     And we'll print it to the screen
    print('Counties in state ' + state + 
          ' have an average population of ' + str(avg))
%%timeit -n 3
 For this method, we start by telling pandas we're interested in grouping by state name, this is the "split"
     And print the results
    print('Counties in state ' + group + 
          ' have an average population of ' + str(avg))
.......................................................................................................................................  
 
 ### Exploratory Data Analysis (EDA)
 ### Basic Data Understanding Univariate And Bivariate Analysis 
 

### Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) is a critical step in the data analysis process. It involves visually and statistically exploring and summarizing the main characteristics, patterns, and relationships present in the dataset. EDA helps data analysts and scientists to gain insights, detect patterns, and identify potential issues or trends in the data. It often serves as a foundation for further data cleaning, preprocessing, and modeling.

Key goals and techniques of Exploratory Data Analysis:

Data Summarization:

Getting an overview of the data by examining its shape (number of rows and columns) and data types.
Computing summary statistics like mean, median, standard deviation, and quartiles to understand the central tendency and variability of the data.
Data Visualization:

Creating various visual representations of the data, such as histograms, bar charts, line plots, scatter plots, and box plots, to reveal underlying patterns, distributions, and relationships.
Using color, size, and shape to encode additional information in visualizations.
Identifying Data Distributions:

Understanding the distribution of numerical features to identify skewness, outliers, or potential transformation needs.
Identifying the frequency of categories in categorical features.
Correlation Analysis:

Studying the correlation between numerical features to identify strong positive or negative relationships.
Using correlation matrices or heatmaps to visualize the correlation between multiple variables.
Handling Missing Data:

Investigating the presence of missing values and understanding their impact on the data and analysis.
Deciding on an appropriate strategy for handling missing data, such as imputation or removal.
Feature Importance:

Assessing the importance of features in relation to the target variable.
Identifying features that are highly predictive or irrelevant for the analysis.
Outlier Detection:

Identifying and examining potential outliers that might skew the analysis or modeling results.
Data Interactions:

Exploring interactions between different features to uncover any hidden patterns or associations.
Geospatial Analysis (if applicable):

For datasets with geospatial information, visualizing data on maps and analyzing spatial patterns.
The EDA process is not a rigid set of steps but a flexible and iterative exploration of the data. Visualizations play a crucial role in EDA, as they provide a powerful way to understand complex relationships and patterns within the data. Python libraries such as Pandas, Matplotlib, Seaborn, and Plotly are commonly used for conducting EDA.

### Basic Data Understanding Univariate And Bivariate Analysis 
Basic Data Understanding, Univariate Analysis, and Bivariate Analysis are fundamental components of Exploratory Data Analysis (EDA) that help analysts gain insights into the dataset's characteristics and relationships between variables.

Basic Data Understanding:

This step involves getting familiar with the dataset, understanding its structure, and knowing the variables it contains.
It includes checking the number of rows and columns in the dataset, identifying the data types of each variable, and exploring the first few rows of the dataset to get a glimpse of the data.
Basic data understanding also involves checking for missing values, duplicates, and potential data quality issues.
Univariate Analysis:

Univariate analysis focuses on understanding and summarizing individual variables in the dataset.
For numerical variables (continuous or discrete), it includes computing summary statistics like mean, median, standard deviation, and quartiles to understand central tendency and dispersion.
For categorical variables, it involves calculating frequencies and proportions for each category to understand the distribution of categorical data.
Visualization techniques like histograms, box plots, bar charts, and pie charts are commonly used to visually represent univariate data.
Bivariate Analysis:

Bivariate analysis examines the relationship between two variables in the dataset.
It helps identify correlations, associations, or patterns between pairs of variables.
For numerical variables, scatter plots and correlation matrices can be used to visualize the relationship between variables.
For categorical variables, cross-tabulations or stacked bar charts can be used to explore the relationship between categories across two variables.
Bivariate analysis is essential for understanding how variables interact and influence each other.
### Code

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

 Load the dataset (for illustration purposes)
data_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(data_url)

### Basic Data Understanding
print(df.info())
print(df.head())

 ### Univariate Analysis - Numerical Variable (e.g., 'total_bill')
print(df['total_bill'].describe())
plt.hist(df['total_bill'], bins=20)
plt.xlabel('Total Bill')
plt.ylabel('Frequency')
plt.show()

 ### Univariate Analysis - Categorical Variable (e.g., 'sex')
print(df['sex'].value_counts())
plt.bar(df['sex'].value_counts().index, df['sex'].value_counts().values)
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

### Bivariate Analysis - Numerical vs. Numerical (e.g., 'total_bill' vs. 'tip')
sns.scatterplot(x='total_bill', y='tip', data=df)
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()

### Bivariate Analysis - Categorical vs. Numerical (e.g., 'sex' vs. 'total_bill')
sns.boxplot(x='sex', y='total_bill', data=df)
plt.xlabel('Sex')
plt.ylabel('Total Bill')
plt.show()
........................................................................................................................................

 ### Types of Charts And Graphs
 ### GGPLOT

 ### Types of Charts And Graphs
 There are various types of charts and graphs, each designed to visualize specific types of data and display different patterns or relationships. Here are some common types of charts and graphs:

Bar Chart:

A bar chart uses rectangular bars to represent data. The length of each bar is proportional to the value it represents.
Bar charts are commonly used to compare categorical data or to show the frequency of different categories.
Histogram:

A histogram is a graphical representation of the distribution of numerical data. It consists of bars grouped into intervals (bins), with each bar representing the frequency or count of data falling into that bin.
Line Chart:

A line chart displays data points connected by straight lines. It is often used to show trends or changes over time.
Line charts are suitable for visualizing continuous data, such as stock prices, temperature variations, etc.
Pie Chart:

A pie chart represents data as slices of a circle, where each slice represents a different category and its size corresponds to the proportion of that category in the whole dataset.
Pie charts are useful for displaying the relative composition of different parts of a whole.
Scatter Plot:

A scatter plot uses individual data points to display the relationship between two numerical variables.
Each data point on the scatter plot represents a single observation, and the x and y coordinates represent the values of the two variables being compared.
Scatter plots are helpful in identifying patterns or correlations between variables.
Box Plot (Box-and-Whisker Plot):

A box plot provides a visual summary of the distribution of numerical data through quartiles, outliers, and the median.
It shows the range of the data, the median (middle value), and the spread of the middle 50% of the data.
Area Chart:

An area chart is similar to a line chart, but the area between the line and the x-axis is filled to represent the accumulated value or the contribution of each data point to the whole.
Bubble Chart:

A bubble chart is a variation of a scatter plot where each data point is represented by a circle (bubble).
The size of the circle (bubble) can represent a third variable, allowing visualization of three dimensions of data.
Heatmap:

A heatmap represents data using color variations in a matrix format.
Heatmaps are useful for visualizing the intensity or density of data points in two-dimensional space.
Stacked Bar Chart:

A stacked bar chart displays multiple sets of data stacked on top of one another, showing the total and individual contributions of each category.
Radar Chart (Spider Chart):

A radar chart displays data on a two-dimensional plane with multiple axes emerging from a common center point, representing different variables.
It is useful for visualizing multivariate data.

# GGPLOT
ggplot is a data visualization package in the R programming language that provides a flexible and powerful way to create high-quality graphics. It is based on the Grammar of Graphics, a conceptual framework for creating graphics by combining different components in a structured and coherent manner.

The ggplot2 package, created by Hadley Wickham, is the most popular and widely used implementation of the Grammar of Graphics in R. It allows users to create complex visualizations by layering different components, such as data, aesthetics, and geometric shapes, to represent various aspects of the data.

Key features of ggplot2:

Layered Grammar: In ggplot2, a plot is built by adding different layers, including data, aesthetics, and geometric shapes. Each layer contributes to the final visual representation of the data.

Data Mapping: ggplot2 allows users to map different variables to aesthetics, such as color, size, shape, and position, which helps in representing additional dimensions of the data.

Wide Range of Geometric Shapes: ggplot2 supports a wide variety of geometric shapes, including points, lines, bars, histograms, boxplots, smooth curves, and more.

Faceting: Faceting enables users to create small multiples or subplots based on one or more categorical variables, which helps in exploring patterns in different subsets of the data.

Customization: ggplot2 offers extensive customization options, allowing users to adjust the appearance of the plot, including labels, titles, themes, axes, legends, and color palettes.

### Code
from ggplot import *
import pandas
Download the gapminder CSV file
http://hwheeler01.github.io/comp150/ggplot/gapminder.csv
gap = pandas.read_csv('gapminder.csv')
 Use DataFrame.info to find out more about a DataFrame
gap.info()
Use DataFrame.head to view the first few rows
print(gap.head())
to see more
print(gap.head(n=20))
to see the last few rows
print(gap.tail())
print(gap.describe())
Let's initialize a plot
ggplot(gap, aes(x = 'gdpPercap', y = 'lifeExp'))
Add points
ggplot(gap, aes(x = 'gdpPercap', y = 'lifeExp')) + geom_point()
 How about some color?
ggplot(gap, aes(x = 'gdpPercap', y = 'lifeExp', color = 'continent')) + geom_point() + scale_x_log()
Plot lifeExp vs. year colored by continent
ggplot(gap, aes(x = 'year', y = 'lifeExp', color = 'continent')) + geom_point()
.........................................................................................................................................

 ### Tools for Data Visualization
 ### Data Visualization


  ### Tools for Data Visualization
  There are several tools available for data visualization, each with its own strengths and capabilities. The choice of tool depends on factors such as the complexity of the data, the level of interactivity required, the type of visualizations needed, and the programming language preference. Here are some popular tools for data visualization:

ggplot2 (R):

ggplot2 is an R package based on the Grammar of Graphics. It allows users to create sophisticated and highly customizable data visualizations using a layered approach.
Ideal for static and publication-quality plots, especially for data analysis in R.
Matplotlib (Python):

Matplotlib is a widely used Python library for data visualization. It provides a flexible and comprehensive set of functions for creating static, interactive, and animated plots.
Suitable for basic to intermediate visualizations in Python.
Seaborn (Python):

Seaborn is built on top of Matplotlib and provides a higher-level interface for creating aesthetically pleasing statistical graphics.
It is particularly useful for visualizing statistical relationships and has many built-in themes for easy customization.
Plotly (Python and JavaScript):

Plotly is a powerful tool for creating interactive and dynamic visualizations in both Python and JavaScript.
It supports a wide range of chart types and is often used for creating web-based interactive dashboards.
Tableau (Desktop and Public):

Tableau is a popular data visualization software that offers a user-friendly drag-and-drop interface for creating visualizations without the need for coding.
Ideal for business analysts and non-technical users to explore data and create interactive dashboards.
Power BI (Microsoft):

Power BI is a business intelligence tool by Microsoft that provides data visualization capabilities along with data modeling, reporting, and dashboard creation.
Suitable for creating interactive and data-driven dashboards.
D3.js (JavaScript):

D3.js is a JavaScript library for creating interactive and data-driven visualizations directly in web browsers.
It provides fine-grained control over visual elements and is commonly used for custom visualizations and infographics.
Bokeh (Python):

Bokeh is a Python library for interactive data visualization that targets modern web browsers. It provides a high-level and concise API for creating interactive plots.
Suitable for creating web-based interactive visualizations.
Excel and Google Sheets:

For quick and simple visualizations, spreadsheet software like Microsoft Excel and Google Sheets offer built-in charting tools for basic visualization needs.
Altair (Python):

Altair is a declarative statistical visualization library for Python that allows users to define visualizations by writing concise and expressive code.
These tools provide a wide range of options for data visualization, catering to various skill levels and project requirements. Depending on the task at hand, data analysts and scientists can choose the tool that best fits their needs and enables them to effectively communicate insights from their data.

 # Data Visualization
Data visualization is a very important part of data analysis. You can use it to explore your data. If you understand your data well, you’ll have a better chance to find some insights. Finally, when you find any insights, you can use visualizations again to be able to share your findings with other people.

For example, look at the nice plot below. This plot shows the Life Expectancy and Income of 182 nations in the year 2015. Each bubble represents a country, the color represents a region, and the size represents the population of that country.
Before we look at some kinds of plots, we’ll introduce some basic rules. Those rules help us make nice and informative plots instead of confusing ones.
# Basic Visualization Rules
The first step is to choose the appropriate plot type. If there are various options, we can try to compare them, and choose the one that fits our model the best.
Second, when we choose your type of plot, one of the most important things is to label your axis. If we don’t do this, the plot is not informative enough. When there are no axis labels, we can try to look at the code to see what data is used and if we’re lucky we’ll understand the plot. But what if we have just the plot as an image? What if we show this plot to your boss who doesn’t know how to make plots in Python?
Third, we can add a title to make our plot more informative.
Fourth, add labels for different categories when needed.
Five, optionally we can add a text or an arrow at interesting data points.
Six, in some cases we can use some sizes and colors of the data to make the plot more informative.
# Types of Visualizations and Examples with Matplotlib
There are many types of visualizations. Some of the most famous are: line plot, scatter plot, histogram, box plot, bar chart, and pie chart. But among so many options how do we choose the right visualization? First, we need to make some exploratory data analysis. After we know the shape of the data, the data types, and other useful statistical information, it will be easier to pick the right visualization type. By the way, when I used the words “plot”, “chart”, and “visualization” I mean the same thing. Here, I found an image for chart suggestion that can be useful.

There are many visualization packages in Python. One of the most famous is Matplotlib. It can be used in Python scripts, the Python and IPython shells, the Jupyter notebook, and web application servers.

Before we jump into the definitions and examples, I want to show you some basic functions of the matplotlib.pyplot subpackage, that we’ll see in the examples below. Here, I am assuming that the matplotlib.pyplot subpackage is imported with an alias plt.

plt.title(“My Title”) will add a title “My Title” to your plot
plt.xlabel(“Year”) will add a label “Year” to your x-axis
plt.ylabel(“Population”) will add a label “Population” to your y-axis
plt.xticks([1, 2, 3, 4, 5]) set the numbers on the x-axis to be 1, 2, 3, 4, 5. We can also pass and labels as a second argument. For, example, if we use this code plt.xticks([1, 2, 3, 4, 5], ["1M", "2M", "3M", "4M", "5M"]), it will set the labels 1M, 2M, 3M, 4M, 5M on the x-axis.
plt.yticks() - works the same as plt.xticks(), but for the y-axis.
Line Plot: a type of plot which displays information as a series of data points called “markers” connected by straight lines. In this type of plot, we need the measurement points to be ordered (typically by their x-axis values). This type of plot is often used to visualize a trend in data over intervals of time - a time series.
To make a line plot with Matplotlib, we call plt.plot(). The first argument is used for the data on the horizontal axis, and the second is used for the data on the vertical axis. This function generates your plot, but it doesn’t display it. To display the plot, we need to call the plt.show() function. This is nice because we might want to add some additional customizations to our plot before we display it. For example, we might want to add labels to the axis and title for the plot.
import matplotlib.pyplot as plt

years = [1983, 1984, 1985, 1986, 1987]
total_populations = [8939007, 8954518, 8960387, 8956741, 8943721]

plt.plot(years, total_populations)
plt.title("Year vs Population in Bulgaria")
plt.xlabel("Year")
plt.ylabel("Total Population")
plt.show()

Scatter plot: this type of plot shows all individual data points. Here, they aren’t connected with lines. Each data point has the value of the x-axis value and the value from the y-axis values. This type of plot can be used to display trends or correlations. In data science, it shows how 2 variables compare.
To make a scatter plot with Matplotlib, we can use the plt.scatter() function. Again, the first argument is used for the data on the horizontal axis, and the second - for the vertical axis.

mport matplotlib.pyplot as plt

temp = [30, 32, 33, 28.5, 35, 29, 29]
ice_creams_count = [100, 115, 115, 75, 125, 79, 89]

plt.scatter(temp, ice_creams_count)
plt.title("Temperature vs. Sold ice creams")
plt.xlabel("Temperature")
plt.ylabel("Sold ice creams count")
plt.show()

Histogram: an accurate representation of the distribution of numeric data. To create a histogram, first, we divide the entire range of values into a series of intervals, and second, we count how many values fall into each interval. The intervals are also called bins. The bins are consecutive and non-overlapping intervals of a variable. They must be adjacent and are often of equal size.
To make a histogram with Matplotlib, we can use the plt.hist() function. The first argument is the numeric data, the second argument is the number of bins. The default value for the bins argument is 10.
We can see from the histogram above that there are:

5 values between 0 and 3
3 values between 3 and 6 (including)
2 values between 6 (excluding) and 9

import matplotlib.pyplot as plt

numbers = [0.1, 0.5, 1, 1.5, 2, 4, 5.5, 6, 8, 9]

plt.hist(numbers, bins = 3)
plt.xlabel("Number")
plt.ylabel("Frequency")
plt.show()

Box plot, also called the box-and-whisker plot: a way to show the distribution of values based on the five-number summary: minimum, first quartile, median, third quartile, and maximum.
The minimum and the maximum are just the min and max values from our data.
The median is the value that separates the higher half of a data from the lower half. It’s calculated by the following steps: order your values, and find the middle one. In a case when our count of values is even, we actually have 2 middle numbers, so the median here is calculated by summing these 2 numbers and divide the sum by 2. For example, if we have the numbers 1, 2, 5, 6, 8, 9, your median will be (5 + 6) / 2 = 5,5.
The first quartile is the median of the data values to the left of the median in our ordered values. For example, if we have the numbers 1, 3, 4, 7, 8, 8, 9, the first quartile is the median from the 1, 3, 4 values, so it’s 3.
The third quartile is the median of the data values to the right of the median in our ordered values. For example, if we use these numbers 1, 3, 4, 7, 8, 8, 9 again, the third quartile is the median from the 8, 8, 9 values, so it’s 8.
I also want to mention one more statistic here. That is the IQR (Interquartile Range). The IQR approximates the amount of spread in the middle 50% of the data. The formula is the third quartile - the first quartile.
This type of plot can also show outliers. An outlier is a data value that lies outside the overall pattern. They are visualized as circles. When we have outliers, the minimum and the maximum are visualized as the min and the max values from the values which aren’t outliers. There are many ways to identify what is an outlier. A commonly used rule says that a value is an outlier if it’s less than the first quartile - 1.5 * IQR or high than the third quartile + 1.5 * IQR.
values = [1, 2, 5, 6, 6, 7, 7, 8, 8, 8, 9, 10, 21]

plt.boxplot(values)
plt.yticks(range(1, 22))
plt.ylabel("Value")
plt.show()

Bar chart: represents categorical data with rectangular bars. Each bar has a height corresponds to the value it represents. It’s useful when we want to compare a given numeric value on different categories. It can also be used with 2 data series, you can find more information here.
To make a bar chart with Maplotlib, we’ll need the plt.bar() function.
 Our data
labels = ["JavaScript", "Java", "Python", "C#"]
usage = [69.8, 45.3, 38.8, 34.4]

 Generating the y positions. Later, we'll use them to replace them with labels.
y_positions = range(len(labels))

 Creating our bar plot
plt.bar(y_positions, usage)
plt.xticks(y_positions, labels)
plt.ylabel("Usage (%)")
plt.title("Programming language usage")
plt.show()

Pie chart: a circular plot, divided into slices to show numerical proportion. They are widely used in the business world. However, many experts recommend to avoid them. The main reason is that it’s difficult to compare the sections of a given pie chart. Also, it’s difficult to compare data across multiple pie charts. In many cases, they can be replaced by a bar chart.

We make angle judgments when we read a pie chart, but we don’t judge angles very well. - Naomi Robbins

As an example, we’ll see pie chart and bar chart on a data that represents percent revenue per given type wine. Look at the pie chart and try to place the slices in order from largest to smallest. Do you have any troubles?
sizes = [25, 20, 45, 10]
labels = ["Cats", "Dogs", "Tigers", "Goats"]

plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.show()
