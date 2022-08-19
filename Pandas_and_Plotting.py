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

sorted_occup_rep = sorted(occup_rep)
FigureB = occup_rep.plot.bar()

# Sorry, I don't have this one

## Exercise 3