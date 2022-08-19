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

