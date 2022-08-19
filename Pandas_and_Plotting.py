import pandas
import seaborn

## Exercise 1
# a) Loading
df = pandas.DataFrame(seaborn.load_dataset("tips"))

# b) Replacing
df = df.replace(["Thur", "Fri", "Sat", "Sun"],["Thursday","Friday","Saturday","Sunday"])

df["day"]

# c) Plotting
figure1 = seaborn.relplot(x="total_bill", y="tip", data = df, hue = "day", col = "sex")
figure1.savefig("output/tips.pdf")


## Exercise 2
# a) Loading

FNAME = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user"

dfoccup = pandas.read_csv(FNAME, sep='|')

# b) Printing first and last entries

dfoccup.tail(n=10)
dfoccup.head(n=25)

# c) Types of columns

dfoccup.dtypes
# User ID and age are integer (numbers); Gender, Occupation a Zip_code are objects

# d) Occupations

occup_rep = dfoccup["occupation"].value_counts()

# e) Counting occupations

occup_num = len(occup_rep)
occup_num_max = occup_rep.idxmax()

# There are 21 different occupations, the most common one is "Student"

# f) Plotting

occup_sort = occup_rep.sort_index()
figure2 = occup_sort.plot.bar(x='Occupation')


# Sorry, I don't have the axes names and the saved file in 2f)

## Exercise 3

# a) Loading, Changing Column Names

FNAME2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = pandas.read_csv(FNAME2, sep=',')

iris.columns = ["sepal length (in cm)", "sepal width (in cm)", "petal length (in cm)", "petal width (in cm)", "class"]

# b) Setting up values to missing

iris.loc[10:29, "petal length (in cm)"] = "NaN"

# c) Replacing missing values

iris["petal length (in cm)"] = iris["petal length (in cm)"].replace(["NaN"],1.0)

# d) Saving CSV

iris.to_csv("output/iris.csv")

# e) Plotting

iris.plot.scatter(x="class", y="sepal length (in cm)")

# Sorry, I don't have multiple plots

## Exercise 4

# a) Loading

FNAME3 = "https://query.data.world/s/wsjbxdqhw6z6izgdxijv5p2lfqh7gx"
mem_df = pandas.read_csv(FNAME3, sep=',')

# b) Inspecting

mem_df.info()
mem_df.info(memory_usage="deep")

# In the second case, there is a way higher memory usage (859.4 MB) compared to the first case (211.2+ MB)

# c) Copy

copy_mem_df = mem_df.select_dtypes(include=['object'])

# d) New object summary

copy_mem_df.describe()

# Those columns that have limited number of selection (day, day/night, acquisition_info - Y/N)

# e) Objects or categories

# Categories are probably more useful in case of columns with limited observations,
# where it makes sense to analyze their aggregations

# f) Object to category

mem_df["completion"].nunique()
# Completion excluded, too much variety
mem_df["forefeit"].nunique()
mem_df["protest"].nunique()
mem_df["v_league"].nunique()
mem_df["h_league"].nunique()

mem_df["day_of_week"] = mem_df["day_of_week"].astype("category")
mem_df["forefeit"] = mem_df["forefeit"].astype("category")
mem_df["protest"] = mem_df["protest"].astype("category")
mem_df["day_night"] = mem_df["day_night"].astype("category")
mem_df["v_league"] = mem_df["v_league"].astype("category")
mem_df["h_league"] = mem_df["h_league"].astype("category")
mem_df["acquisition_info"] = mem_df["acquisition_info"].astype("category")

# g) Memory size

mem_df.dtypes
mem_df.info()
mem_df.info(memory_usage="deep")

# The memory size decreased: 803.4 MB is the second case (compared to 859.4 MB previously)
# and 203.1+MB in the first case (vs. 211.2+ MB). This should be another reason why to change
# some appropriate objects to categories

# h)

# i) Numerical only

mem_df_num = mem_df.select_dtypes("number")

mem_df_num.to_csv("output/mem.csv")
mem_df_num.to_feather("output/mem.feather")

# The CSV file has approximately 51 MB, while the feather file has approx. 21 MB
# The feather files are significantly smaller (by 30 MB in this case), probably due to the file format (Arrow IPC) and compression