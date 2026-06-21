
import pandas as pd
import numpy as np
import pickle
import os

# 1. DEFINICIÓN DE LA CLASE DEL ENSAMBLE (Crucial para pickle)
class EnsembleSoccerModel:
    def __init__(self, xgb_m, cat_m, nn_m, lstm_m, scaler_m):
        self.xgb = xgb_m
        self.cat = cat_m
        self.nn = nn_m
        self.lstm = lstm_m
        self.scaler = scaler_m

    def predict_proba(self, X):
        p_xgb = self.xgb.predict_proba(X)
        p_cat = self.cat.predict_proba(X)
        X_s = self.scaler.transform(X)
        p_nn = self.nn.predict(X_s, verbose=0)
        X_l = X_s.reshape((X_s.shape[0], 1, X_s.shape[1]))
        p_lstm = self.lstm.predict(X_l, verbose=0)
        return (p_xgb * 0.35) + (p_cat * 0.35) + (p_nn * 0.15) + (p_lstm * 0.15)

# Configuración de Columnas
COLUMNAS_MODELO = [
    'xg_favor_local', 'xg_contra_local', 'xg_favor_visita', 'xg_contra_visita',
    'ranking_fifa_local', 'ranking_fifa_visita', 'dias_descanso_local', 'dias_descanso_visita'
]

def ejecutar_motor_prediccion():
    try:
        # 2. Cargar el cerebro del Superagente
        if not os.path.exists('modelo_xgboost.pkl'):
            print("❌ Error: modelo_xgboost.pkl no encontrado.")
            return
            
        with open('modelo_xgboost.pkl', 'rb') as f:
            ensemble_model = pickle.load(f)

        # 3. Cargar Datos Oficiales (Verificar existencia)
        files = ['wc_2026_fixtures.csv', 'fifa_ranking_2026-06-08.csv']
        for f_check in files:
            if not os.path.exists(f_check):
                print(f"❌ Error: {f_check} no encontrado.")
                return

        df_fixtures = pd.read_csv('wc_2026_fixtures.csv')
        df_rankings = pd.read_csv('fifa_ranking_2026-06-08.csv')

        rank_dict = df_rankings.set_index('team')['rank'].to_dict()
        promedios_xg = {
            'xg_favor_local': 1.6, 'xg_contra_local': 1.2,
            'xg_favor_visita': 1.4, 'xg_contra_visita': 1.5
        }

        print("🚀 MOTOR DE PREDICCIÓN FIFA 2026 INICIADO")
        print("------------------------------------------")

        for _, row in df_fixtures.head(5).iterrows():
            local, visita = row['team1'].strip(), row['team2'].strip()
            r_l = rank_dict.get(local, 50)
            r_v = rank_dict.get(visita, 50)

            input_data = pd.DataFrame([[
                promedios_xg['xg_favor_local'], promedios_xg['xg_contra_local'],
                promedios_xg['xg_favor_visita'], promedios_xg['xg_contra_visita'],
                r_l, r_v, 5, 5
            ]], columns=COLUMNAS_MODELO)

            probs = ensemble_model.predict_proba(input_data)[0]

            print(f"⚽ {local} vs {visita}")
            print(f"   L: {probs[0]:.1%} | E: {probs[1]:.1%} | V: {probs[2]:.1%}")
            print("------------------------------------------")

    except Exception as e:
        print(f"❌ Error en el motor: {e}")

if __name__ == '__main__':
    ejecutar_motor_prediccion()
