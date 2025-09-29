# test_complete_system.py
import requests
import json
import time

def test_complete_system():
    print("🧪 Probando sistema completo de predicciones...")
    
    base_url = "http://localhost:8000"
    
    # 1. Health check
    print("\n1️⃣ Verificando salud del backend...")
    try:
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Backend funcionando correctamente")
            print(f"   Response: {health_response.json()}")
        else:
            print(f"❌ Backend no responde correctamente: {health_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error conectando al backend: {e}")
        return
    
    # 2. Test prediction
    print("\n2️⃣ Probando predicción...")
    try:
        prediction_data = {
            "records": [
                {
                    "feature_1": 10.5,
                    "feature_2": 20.1,
                    "feature_3": 5.0,
                    "feature_4": 15.2,
                    "feature_5": 8.7
                }
            ]
        }
        
        predict_response = requests.post(f"{base_url}/predict", json=prediction_data, timeout=30)
        if predict_response.status_code == 200:
            print("✅ Predicción exitosa")
            result = predict_response.json()
            print(f"   Model version: {result.get('model_version')}")
            print(f"   Results count: {len(result.get('results', []))}")
        else:
            print(f"❌ Error en predicción: {predict_response.status_code}")
            print(f"   Error: {predict_response.text}")
            return
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return
    
    # 3. Test predictions endpoint (sin autenticación por ahora)
    print("\n3️⃣ Probando endpoint de predicciones...")
    try:
        # Como no tenemos autenticación configurada, vamos a probar directamente la base de datos
        print("   (Saltando endpoint de predicciones por falta de autenticación)")
    except Exception as e:
        print(f"❌ Error en endpoint de predicciones: {e}")
    
    # 4. Test frontend
    print("\n4️⃣ Verificando frontend...")
    try:
        frontend_response = requests.get("http://localhost:3000", timeout=5)
        if frontend_response.status_code == 200:
            print("✅ Frontend funcionando correctamente")
        else:
            print(f"❌ Frontend no responde: {frontend_response.status_code}")
    except Exception as e:
        print(f"❌ Error conectando al frontend: {e}")
    
    print("\n🎉 Prueba del sistema completada!")

if __name__ == "__main__":
    test_complete_system()
