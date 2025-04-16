import pandas as pd
from IPython.core.display_functions import display
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

Data = pd.read_csv("titanic.csv")
Survived = pd.read_csv("Survive.csv")

Sex = LabelEncoder()
Data["Sex"] = Sex.fit_transform(Data["Sex"])

y = Data["Survived"]
x = Data.drop(columns=["Name", "Fare", "Survived"])

x_training, x_test, y_training, y_test = train_test_split(x, y, test_size=0.3)

Tree = RandomForestClassifier(random_state=42)

Tree.fit(x_training, y_training)

Survived = Survived.drop(columns = ["Name", "Fare"])
Survived["Sex"] = Sex.fit_transform(Survived["Sex"])

Decision = Tree.predict(Survived)

Decision_convert = np.where(Decision == 1, "Survived", "Died")

display(Decision_convert)







