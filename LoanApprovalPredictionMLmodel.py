import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


#Read csv file
loan_approval=pd.read_csv('C:/Users/DELL/Desktop/ML learning internship/loan approval task2 2/dataset/archive/loan_approval_dataset.csv')

#Show first five rows
print(loan_approval.head())

#Explore data
print("how much null in every column: ",loan_approval.isnull().sum())
print("columns info: ",loan_approval.info())
print("statistical information about columns: \n",loan_approval.describe())
print("how much duplicated rows? \n",loan_approval.duplicated().sum())
print("shape: ", loan_approval.shape)

loan_approval1=loan_approval.copy()
loan_approval1.drop('loan_id',axis=1,inplace=True)

loan_approval1.columns = loan_approval1.columns.str.strip()


#Encode
le = LabelEncoder()
loan_approval1['education'] = le.fit_transform(loan_approval1['education'])
loan_approval1['self_employed'] = le.fit_transform(loan_approval1['self_employed'])
loan_approval1['loan_status'] = le.fit_transform(loan_approval1['loan_status'])

#Features
features_input=['no_of_dependents','education','income_annum','loan_amount','loan_term','cibil_score','residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']

X=loan_approval1[features_input]
Y=loan_approval1['loan_status']

#Encode
X=pd.get_dummies(X,drop_first=True)

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

#Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#OverSampling
os=SMOTE(random_state=42)
X_resampled, y_resampled = os.fit_resample(X_train, y_train)
oversampled_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['loan_status'])], axis=1)

print(Y.value_counts())

#Building model
model = LogisticRegression()
model.fit(X_resampled, y_resampled)

#Predictions
y_pred = model.predict(X_test)

#Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



