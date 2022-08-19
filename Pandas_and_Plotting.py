import pandas
import seaborn

df = pandas.DataFrame(seaborn.load_dataset("tips"))

df = df.replace(["Thur", "Fri", "Sat", "Sun"],["Thursday","Friday","Saturday","Sunday"])

df["day"]

figure1 = seaborn.relplot(x="total_bill", y="tip", data = df, hue = "day", col = "sex")
figure1.savefig("output/tips.pdf")
