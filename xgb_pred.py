import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

data1=pd.read_csv('material_with_expanded_applications.csv')
df = pd.DataFrame(data1)

y = data1['Application']
X = data1.drop(columns=['Use','Material', 'Application']) 

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y = y_encoded

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8 ,random_state=4)

xgb_model = XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_), random_state=8)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


def getApplication(float_Su, float_Sy, float_E, float_G, float_mu, float_Ro):
    row = {'Su': float_Su, 'Sy': float_Sy, 'E': float_E, 'G': float_G, 'mu': float_mu, 'Ro': float_Ro}
    input_data = pd.DataFrame([row])
    predicted_label = xgb_model.predict(input_data)
    decoded_label = label_encoder.inverse_transform(predicted_label)
    return decoded_label[0]
