
import pandas as pd
import numpy as np
import pickle
import os

# Configuración de Columnas
COLUMNAS_MODELO = [
    'xg_favor_local', 'xg_contra_local', 'xg_favor_visita', 'xg_contra_visita',
    'ranking_fifa_local', 'ranking_fifa_visita', 'dias_descanso_local', 'dias_descanso_visita'
]

def ejecutar_motor_prediccion():
    try:
        # 1. Cargar el cerebro del Superagente
        with open('modelo_xgboost.pkl', 'rb') as f:
            ensemble_model = pickle.load(f)
        
        # 2. Cargar Datos Oficiales
        df_fixtures = pd.read_csv('wc_2026_fixtures.csv')
        df_rankings = pd.read_csv('fifa_ranking_2026-06-08.csv')
        df_historial = pd.read_csv('wc_all_matches.csv')
        
        rank_dict = df_rankings.set_index('team')['rank'].to_dict()
        # Calcular promedios reales para features base
        # (Mapeo simplificado para el script de produccion)
        promedios_xg = {
            'xg_favor_local': 1.6, 
            'xg_contra_local': 1.2,
            'xg_favor_visita': 1.4,
            'xg_contra_visita': 1.5
        }

        print("🚀 MOTOR DE PREDICCIÓN FIFA 2026 INICIADO")
        print("------------------------------------------")

        # 3. Predicción de los próximos partidos (Muestra)
        for _, row in df_fixtures.head(5).iterrows():
            local, visita = row['team1'].strip(), row['team2'].strip()
            r_l = rank_dict.get(local, 50)
            r_v = rank_dict.get(visita, 50)

            # Construir vector de entrada
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
