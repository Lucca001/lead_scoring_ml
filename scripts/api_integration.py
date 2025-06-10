from hubspot import HubSpot
from hubspot.crm.contacts import ApiException
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler


api_key = 'API_TREINO_EXEMPLO_PRIVADA_HUBSPOT'
client = HubSpot(api_key=api_key)


model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def get_leads_from_hubspot():
    all_leads = []
    try:
        response = client.crm.contacts.basic_api.get_page(limit=100)
        for contact in response.results:
            lead = {
                "email": contact.properties.get("email"),
                "company": contact.properties.get("company"),
                "jobtitle": contact.properties.get("jobtitle"),
                "createdate": contact.properties.get("createdate"),
                "lifecyclestage": contact.properties.get("lifecyclestage"),
                "tempo_site": 4.5,            # Simulados (até integrar real)
                "interacoes": 3,
                "valor_proposta": 10000,
                "setor_Varejo": 1,
                "setor_Educação": 0,
                "setor_Saúde": 0,
                "cargo_Gerente": 1,
                "cargo_Diretor": 0,
                "canal_Indicação": 0,
                "canal_Pago": 1
            }
            all_leads.append(lead)
    except ApiException as e:
        print(f"Erro ao acessar HubSpot: {e}")
    return pd.DataFrame(all_leads)

def aplicar_modelo(df):
    features = [
        "tempo_site", "interacoes", "valor_proposta",
        "setor_Varejo", "setor_Educação", "setor_Saúde",
        "cargo_Gerente", "cargo_Diretor",
        "canal_Indicação", "canal_Pago"
    ]
    X = df[features]
    X_scaled = scaler.transform(X)
    df['score_conversao'] = model.predict_proba(X_scaled)[:, 1]
    return df[['email', 'score_conversao']]

if __name__ == "__main__":
    df_leads = get_leads_from_hubspot()
    df_resultado = aplicar_modelo(df_leads)
    print(df_resultado.head())

  
    df_resultado.to_csv("output/predicoes_leads.csv", index=False)
