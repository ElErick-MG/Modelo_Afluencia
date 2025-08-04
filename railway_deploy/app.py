from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Variables globales para cargar el modelo una sola vez
modelo = None
features_info = None
patrones = None

def cargar_modelo():
    """Cargar modelo y recursos al iniciar la aplicación"""
    global modelo, features_info, patrones
    
    try:
        # Cargar modelo principal
        modelo = joblib.load('modelo_api_export/modelo_turismo_ecuador.pkl')
        print("✅ Modelo cargado exitosamente")
        
        # Cargar información de features
        with open('modelo_api_export/features_modelo.json', 'r', encoding='utf-8') as f:
            features_info = json.load(f)
        print("✅ Features cargadas exitosamente")
        
        # Cargar patrones de corrección
        with open('modelo_api_export/patrones_correccion.json', 'r', encoding='utf-8') as f:
            patrones = json.load(f)
        print("✅ Patrones de corrección cargados exitosamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error cargando recursos: {str(e)}")
        return False

def aplicar_correccion_avanzada(prediccion_base, provincia, dia_semana, es_vacaciones=0, mes=1):
    """Aplicar correcciones avanzadas según patrones específicos"""
    try:
        # Factor base
        factor_total = 1.0
        factores_aplicados = {
            'provincia': 1.0,
            'dia_semana': 1.0,
            'vacaciones': 1.0,
            'estacional': 1.0
        }
        
        # Corrección por provincia (si existe en los patrones)
        provincia_upper = provincia.upper()
        if provincia_upper in patrones['PATRONES_PROVINCIA']:
            factor_provincia = patrones['PATRONES_PROVINCIA'][provincia_upper].get('factor_base', 1.0)
            factores_aplicados['provincia'] = factor_provincia
            factor_total *= factor_provincia
        
        # Corrección por día de la semana
        if str(dia_semana) in patrones['PATRONES_DIA_SEMANA']:
            factor_dia = patrones['PATRONES_DIA_SEMANA'][str(dia_semana)].get('factor', 1.0)
            factores_aplicados['dia_semana'] = factor_dia
            factor_total *= factor_dia
        
        # Corrección por vacaciones
        if es_vacaciones:
            factor_vacaciones = 1.15  # Incremento del 15% en vacaciones
            factores_aplicados['vacaciones'] = factor_vacaciones
            factor_total *= factor_vacaciones
        
        # Corrección estacional por mes
        factores_estacionales = {
            1: 1.1,   # Enero - temporada alta
            2: 1.05,  # Febrero
            3: 0.95,  # Marzo
            4: 0.9,   # Abril
            5: 0.9,   # Mayo
            6: 1.2,   # Junio - vacaciones escolares
            7: 1.25,  # Julio - temporada alta
            8: 1.3,   # Agosto - temporada alta
            9: 0.9,   # Septiembre
            10: 0.95, # Octubre
            11: 1.0,  # Noviembre
            12: 1.15  # Diciembre - fiestas
        }
        
        factor_estacional = factores_estacionales.get(mes, 1.0)
        factores_aplicados['estacional'] = factor_estacional
        factor_total *= factor_estacional
        
        # Aplicar corrección
        prediccion_corregida = prediccion_base * factor_total
        
        return {
            'prediccion_corregida': prediccion_corregida,
            'factores_aplicados': factores_aplicados,
            'factor_total': factor_total
        }
        
    except Exception as e:
        print(f"Error en corrección: {str(e)}")
        return {
            'prediccion_corregida': prediccion_base,
            'factores_aplicados': {'error': str(e)},
            'factor_total': 1.0
        }

def categorizar_afluencia(afluencia):
    """Categorizar el nivel de afluencia"""
    if afluencia >= 35:
        return {
            'categoria': 'MUY ALTA',
            'emoji': '🔥🔥🔥',
            'recomendacion': 'Excelente día para turismo - alta demanda'
        }
    elif afluencia >= 25:
        return {
            'categoria': 'ALTA',
            'emoji': '🔥🔥',
            'recomendacion': 'Muy buen día para actividades turísticas'
        }
    elif afluencia >= 15:
        return {
            'categoria': 'MEDIA',
            'emoji': '🔥',
            'recomendacion': 'Día moderado para turismo'
        }
    else:
        return {
            'categoria': 'BAJA',
            'emoji': '❄️',
            'recomendacion': 'Día tranquilo, menos turismo'
        }

@app.route('/', methods=['GET'])
def home():
    """Endpoint principal con información de la API"""
    return jsonify({
        'mensaje': '🇪🇨 API de Predicción de Afluencia Turística - Ecuador',
        'version': '1.0',
        'autor': 'Sistema de Predicción Turística',
        'endpoints': {
            '/predict': 'POST - Realizar predicción de afluencia',
            '/health': 'GET - Estado de salud de la API',
            '/features': 'GET - Lista de features requeridas'
        },
        'status': 'activo'
    })

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de salud"""
    modelo_cargado = modelo is not None
    features_cargadas = features_info is not None
    patrones_cargados = patrones is not None
    
    return jsonify({
        'status': 'healthy' if all([modelo_cargado, features_cargadas, patrones_cargados]) else 'error',
        'modelo_cargado': modelo_cargado,
        'features_cargadas': features_cargadas,
        'patrones_cargados': patrones_cargados,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Obtener lista de features requeridas"""
    if features_info is None:
        return jsonify({'error': 'Features no cargadas'}), 500
    
    return jsonify(features_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal de predicción"""
    try:
        # Verificar que el modelo esté cargado
        if modelo is None or features_info is None:
            return jsonify({'error': 'Modelo no cargado correctamente'}), 500
        
        # Obtener datos del request
        data = request.json
        if not data:
            return jsonify({'error': 'No se enviaron datos'}), 400
        
        # Crear DataFrame con las features necesarias
        df_input = pd.DataFrame([data])
        
        # Asegurar que tiene todas las columnas necesarias
        for feature in features_info['features']:
            if feature not in df_input.columns:
                df_input[feature] = 0  # Valor por defecto
        
        # Ordenar columnas según el modelo entrenado
        df_input = df_input[features_info['features']]
        
        # Hacer predicción base
        prediccion_base = float(modelo.predict(df_input)[0])
        
        # Obtener parámetros para corrección
        provincia = data.get('provincia', 'PICHINCHA')
        dia_semana = data.get('dia_semana', 1)
        es_vacaciones = data.get('Es_Vacaciones', 0)
        mes = data.get('Mes', 1)
        
        # Aplicar correcciones avanzadas
        resultado_correccion = aplicar_correccion_avanzada(
            prediccion_base, 
            provincia, 
            dia_semana, 
            es_vacaciones, 
            mes
        )
        
        # Categorizar resultado
        afluencia_final = resultado_correccion['prediccion_corregida']
        categoria_info = categorizar_afluencia(afluencia_final)
        
        # Respuesta completa
        respuesta = {
            'prediccion': {
                'afluencia': round(afluencia_final, 2),
                'prediccion_base': round(prediccion_base, 2),
                'categoria': categoria_info['categoria'],
                'emoji': categoria_info['emoji'],
                'recomendacion': categoria_info['recomendacion']
            },
            'detalles': {
                'provincia': provincia.upper(),
                'dia_semana': dia_semana,
                'mes': mes,
                'es_vacaciones': bool(es_vacaciones),
                'factores_aplicados': resultado_correccion['factores_aplicados'],
                'factor_total': round(resultado_correccion['factor_total'], 3)
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'modelo': 'RandomForest - Turismo Ecuador v1.0',
                'status': 'success'
            }
        }
        
        return jsonify(respuesta)
        
    except Exception as e:
        return jsonify({
            'error': f'Error en predicción: {str(e)}',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 400

@app.route('/predict/simple', methods=['POST'])
def predict_simple():
    """Endpoint simplificado para N8N"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Crear DataFrame
        df_input = pd.DataFrame([data])
        
        # Completar features faltantes
        for feature in features_info['features']:
            if feature not in df_input.columns:
                df_input[feature] = 0
        
        df_input = df_input[features_info['features']]
        
        # Predicción
        prediccion = float(modelo.predict(df_input)[0])
        
        # Aplicar corrección básica
        provincia = data.get('provincia', 'PICHINCHA')
        dia_semana = data.get('dia_semana', 1)
        mes = data.get('Mes', 1)
        
        resultado = aplicar_correccion_avanzada(prediccion, provincia, dia_semana, 0, mes)
        afluencia = round(resultado['prediccion_corregida'], 2)
        
        # Respuesta simple
        return jsonify({
            'afluencia': afluencia,
            'categoria': categorizar_afluencia(afluencia)['categoria'],
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 400

# Cargar modelo al iniciar
if __name__ == '__main__':
    print("🚀 Iniciando API de Predicción Turística...")
    
    if cargar_modelo():
        print("✅ Todos los recursos cargados correctamente")
        
        # Obtener puerto de Railway o usar 5000 por defecto
        port = int(os.environ.get('PORT', 5000))
        
        print(f"🌐 Servidor iniciando en puerto {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("❌ Error cargando recursos. No se puede iniciar el servidor.")
else:
    # Para producción (cuando se ejecuta con gunicorn)
    cargar_modelo()
