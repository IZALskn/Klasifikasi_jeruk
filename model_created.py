import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


df = pd.read_csv("data_jeruk.csv")
print(df, '\n')
print(df.shape, '\n')
print(df.columns, '\n')
print(df.info(), '\n')
print(df.describe(), '\n')
print(df["asal_daerah"].value_counts(), '\n')
print(df["warna"].value_counts(), '\n')
print(df["musim_panen"].value_counts(), '\n')
print(df["kualitas"].value_counts(), '\n')
print(df.head(), '\n')
print(df.tail(), '\n')

#================================================================================================================================


bagus = df[df['kualitas']=='Bagus'] 
sedang = df[df['kualitas']=='Sedang']
jelek = df[df['kualitas']=='Jelek']

plt.figure(figsize=(6,5))

plt.scatter(bagus['diameter'], bagus['berat'], s=100, alpha=0.7, color='green', label='Bagus')
plt.scatter(sedang['diameter'], sedang['berat'], s=100, alpha=0.7, color='yellow', label='Sedang')
plt.scatter(jelek['diameter'], jelek['berat'], s=100, alpha=0.7, color='red', label='Jelek')

plt.xlabel('Diameter')
plt.ylabel('Berat')
plt.title('Diameter vs Berat')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

plt.savefig('diameter&berat.png')
plt.close()
print('berhasil disimpan!!')

#---------------------------------------------------

plt.figure(figsize=(6,5))

plt.scatter(bagus['tebal_kulit'], bagus['kadar_gula'], s=100, alpha=0.7, color='green', label='Bagus')
plt.scatter(sedang['tebal_kulit'], sedang['kadar_gula'], s=100, alpha=0.7, color='yellow', label='Sedang')
plt.scatter(jelek['tebal_kulit'], jelek['kadar_gula'], s=100, alpha=0.7, color='red', label='Jelek')

plt.xlabel('Tebal Kulit')
plt.ylabel('Kadar gula')
plt.title('Tebal kulit vs Kadar gula')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

plt.savefig('tebal_kulit&Kadar_gula.png')
plt.close()
print('berhasil disimpan!!')

#=====================================================


X = df[['diameter', 'berat', 'tebal_kulit', 'kadar_gula',
        'asal_daerah', 'warna', 'musim_panen']]
y = df['kualitas']

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42
  )
  
numeric_columns = ['diameter', 'berat', 'tebal_kulit', 'kadar_gula']
categorical_columns = ['asal_daerah', 'musim_panen']
ordinal_columns = ['warna']
  
warna_order = ['hijau', 'kuning', 'oranye']
ordinal_order = [warna_order]
  
preprocessing = ColumnTransformer(
  transformers = [
    ('scaler', StandardScaler(), numeric_columns),
    ('ohe', OneHotEncoder(), categorical_columns),
    ('oe', OrdinalEncoder(categories=ordinal_order), ordinal_columns)
    ]
  )
    
model = Pipeline(
  steps=[
    ('preprocessing', preprocessing),
    ('model', LogisticRegression())
    ]
  )
      
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy :', accuracy_score(y_test, y_pred))
print('\n Classificaton Report :', classification_report(y_test, y_pred))
print('\n Confusion', confusion_matrix(y_test, y_pred))

data_baru = pd.DataFrame([[7.82, 160.5, 1, 13, 'Kalimantan', 'kuning', 'kemarau']], columns=['diameter', 'berat', 'tebal_kulit', 'kadar_gula', 'asal_daerah', 'warna', 'musim_panen'])

prediksi = model.predict(data_baru)[0]
presentase = max(model.predict_proba(data_baru)[0])
print(f'model memprediksi:{prediksi} dengan tingkat keyakinan {presentase*100:.2f}%')

#================================================================== Penhimpanan Model

joblib.dump(model, 'model_klasifikasi_jeruk.joblib')