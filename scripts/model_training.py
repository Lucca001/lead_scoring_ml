import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Simulando dataset
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'setor': np.random.choice(['Tecnologia', 'Varejo', 'Educação', 'Saúde'], n),
    'cargo': np.random.choice(['Analista', 'Gerente', 'Diretor'], n),
    'canal': np.random.choice(['Orgânico', 'Pago', 'Indicação'], n),
    'tempo_site': np.random.normal(3, 1, n).round(1),
    'interacoes': np.random.poisson(2, n),
    'valor_proposta': np.random.uniform(1000, 20000, n).round(2),
    'converteu': np.random.choice([0, 1], n, p=[0.7, 0.3])
})

# 2. Pré-processamento
df_encoded = pd.get_dummies(df.drop('converteu', axis=1), drop_first=True)
X = df_encoded
y = df['converteu']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Escalonamento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Avaliação
y_pred = model.predict(X_test_scaled)
print(\"Relatório de Classificação:\")
print(classification_report(y_test, y_pred))
print(\"AUC-ROC:\", roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))

# 7. Importância das variáveis
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh', title='Top Variáveis (Importância)')
plt.tight_layout()
plt.show()

# 8. Predição de lead exemplo
lead_exemplo = pd.DataFrame({
    'tempo_site': [4.2],
    'interacoes': [3],
    'valor_proposta': [12000],
    'setor_Varejo': [1],
    'setor_Educação': [0],
    'setor_Saúde': [0],
    'cargo_Gerente': [1],
    'cargo_Diretor': [0],
    'canal_Indicação': [0],
    'canal_Pago': [1]
})
lead_exemplo_scaled = scaler.transform(lead_exemplo)
prob = model.predict_proba(lead_exemplo_scaled)[0][1]
print(f\"Chance de conversão do lead: {prob:.2%}\")
