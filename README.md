# 📈 Lead Scoring com Machine Learning Supervisionado

Projeto de classificação binária com foco comercial: prever a probabilidade de um lead se converter (virar cliente) com base em dados simulados de CRM.

---

## 🎯 Objetivo

Criar um modelo supervisionado (Random Forest) para priorização de leads, permitindo que times de marketing e vendas foquem nos contatos com maior probabilidade de conversão.

---

## 📂 Estrutura do projeto


---

## ⚙️ Tecnologias utilizadas

- Python 3.10+
- Pandas, NumPy
- Scikit-learn (Machine Learning)
- Matplotlib & Seaborn (visualização)

---

## 📊 Dados

Os dados foram **simulados** para representar situações reais de CRM:

- `setor`, `cargo`, `canal`: variáveis categóricas do lead
- `tempo_site`, `interacoes`, `valor_proposta`: comportamentais
- `converteu`: variável-alvo binária (0 ou 1)

---

## 🧠 Modelo

- Modelo: `RandomForestClassifier`
- Treinamento com `train_test_split` e `StandardScaler`
- Avaliação: `classification_report` e AUC-ROC
- Feature Importance plot para explicar decisões do modelo

---

## 🧪 Exemplo de uso

Você pode prever a chance de conversão de um novo lead assim:

```python
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
