# Scikit Learn
## Logistic Regression
Is used for classification problems, outputs probabilities if > 0.5 the data is labeled 1 else it is labeled 0.
Produces a linear decision boundary.

```
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logreg = LogisticRegression()

X_train, X_text, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

logreg.fit(X_train,y_train)

y_pred= logreg.predict(X_test)
```

## ROC Curve
Stands for Receiver Operating Characteristics curve

```
from sklearn.metrics import roc_curve

# Returns the probability of a given sample being in a particular class (in this case class 1, as is the 2nd column)
y_pred_prob = logreg.predict_proba(X_test)[:,1]

false_positives_rate, true_positives_rate, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot(fpr,tpr, label = 'Logistic Regression')
plt.xlabel(False Positive Rate')
plt.ylabel(True Positive Rate')
plt.titel('Log Reg ROC Curve')
plt.show()
```

