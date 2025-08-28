import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("https://www.stat4decision.com/telecom.csv")

print("----------------COLUMNS----------------------")
print(data.columns)
print("----------------HEAD----------------------")
print(data.head())


data["Churn?"] = data["Churn?"].astype('category')

# on définit x et y
y = data["Churn?"].cat.codes
# on ne prend que les colonnes quantitatives
x = data.select_dtypes(np.number).drop(["Account Length",
                                        "Area Code"], axis=1)

# Scikit-learn décide par défaut d’appliquer une régularisation sur le modèle.
# Ceci s’explique par l’objectif prédictif du machine learning
# mais ceci peut poser des problèmes si votre objectif est de comparer différents outils
# et leurs résultats (notamment R, SAS…)
modele_logit = LogisticRegression(penalty=None, solver='newton-cg')
modele_logit.fit(x, y)


# Si on veut comprendre les coefficients du modèle, scikit-learn stocke les informations dans .coef_,
# nous allons les afficher de manière plus agréable dans un DataFrame avec la constante du modèle
pd.DataFrame(np.concatenate([modele_logit.intercept_.reshape(-1, 1), modele_logit.coef_], axis=1),
             index=["coef"],
             columns=["constante"] + list(x.columns)).T()

# Statsmodels décide par défaut qu’il n’y a pas de constante,
# il faut ajouter donc une colonne dans les données pour la constante, on utilise pour cela un outil de statsmodels
# on ajoute une colonne pour la constante
x_stat = sm.add_constant(x)
# on ajuste le modèle
model = sm.Logit(y, x_stat)
result = model.fit()

# Une autre source d’erreur vient du fait que la classe Logit attend en premier les variables nommées endogènes
# (qu’on désire expliquer donc le y) et ensuite les variables exogènes (qui expliquent y donc le x).
# Cette approche est inversée par rapport à scikit-learn

print(result.summary())
