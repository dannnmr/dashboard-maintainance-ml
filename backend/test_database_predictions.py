#!/usr/bin/env python3
"""
Script para probar el nuevo flujo de predicciones basado en base de datos
"""

import requests
import json
import time
from datetime import datetime

# Configuración
BASE_URL = "http://localhost:8000"
LOGIN_URL = f"{BASE_URL}/auth/login"
MAINTENANCE_URL = f"{BASE_URL}/predictions/maintenance/results"

def test_login():
    """Probar login para obtener token"""
    print("🔐 Probando login...")
    
    login_data = {
        "email": "admin@example.com",
        "password": "admin123"
    }
    
    try:
        response = requests.post(LOGIN_URL, json=login_data)
        if response.status_code == 200:
            token_data = response.json()
            print(f"✅ Login exitoso: {token_data['access_token'][:20]}...")
            return token_data['access_token']
        else:
            print(f"❌ Error en login: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return None

def test_maintenance_results(token):
    """Probar endpoint de maintenance/results"""
    print("\n📊 Probando maintenance/results...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        start_time = time.time()
        response = requests.get(MAINTENANCE_URL, headers=headers)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Respuesta exitosa en {end_time - start_time:.2f}s")
            print(f"📈 Modelo: {data['model_version']}")
            print(f"🔮 Predicciones: {len(data['results'])}")
            
            if data['results']:
                result = data['results'][0]
                print(f"   - Score: {result['score']}")
                print(f"   - Label: {result['label']}")
                print(f"   - Horizon: {result['horizon_shift']} horas")
                print(f"   - Estado Futuro: {result['predicted_future_state']}")
            
            if data.get('data_info'):
                info = data['data_info']
                print(f"📊 Info: {info.get('file_source', 'N/A')}")
                print(f"🎯 Horizonte: {info.get('horizon_shift', 'N/A')} horas")
            
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False

def test_multiple_requests(token):
    """Probar múltiples requests para verificar cache de BD"""
    print("\n🔄 Probando múltiples requests (debería usar cache de BD)...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    times = []
    for i in range(3):
        start_time = time.time()
        response = requests.get(MAINTENANCE_URL, headers=headers)
        end_time = time.time()
        
        times.append(end_time - start_time)
        print(f"   Request {i+1}: {end_time - start_time:.2f}s - Status: {response.status_code}")
    
    avg_time = sum(times) / len(times)
    print(f"📊 Tiempo promedio: {avg_time:.2f}s")
    
    # Los requests subsecuentes deberían ser más rápidos (cache de BD)
    if len(times) > 1 and times[1] < times[0]:
        print("✅ Cache de BD funcionando correctamente")
    else:
        print("⚠️  Cache de BD podría no estar funcionando")

def test_prediction_history(token):
    """Probar historial de predicciones"""
    print("\n📚 Probando historial de predicciones...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    history_url = f"{BASE_URL}/predictions/history?equipment_id=TR01&limit=5"
    
    try:
        response = requests.get(history_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Historial obtenido: {len(data)} predicciones")
            for i, pred in enumerate(data):
                print(f"   {i+1}. ID: {pred['id']}, Status: {pred['status']}, Fecha: {pred['created_at']}")
        else:
            print(f"❌ Error en historial: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")

def main():
    """Función principal de prueba"""
    print("🚀 Iniciando pruebas del sistema de predicciones con BD")
    print("=" * 60)
    
    # 1. Login
    token = test_login()
    if not token:
        print("❌ No se pudo obtener token. Abortando pruebas.")
        return
    
    # 2. Test maintenance results
    success = test_maintenance_results(token)
    if not success:
        print("❌ No se pudo obtener resultados. Abortando pruebas.")
        return
    
    # 3. Test multiple requests
    test_multiple_requests(token)
    
    # 4. Test history
    test_prediction_history(token)
    
    print("\n" + "=" * 60)
    print("✅ Pruebas completadas")
    print("\n💡 El sistema ahora:")
    print("   - Guarda predicciones en base de datos")
    print("   - Usa cache de BD para requests subsecuentes")
    print("   - Mantiene historial de predicciones")
    print("   - Proporciona mejor rendimiento")

if __name__ == "__main__":
    main()
