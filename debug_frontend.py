# debug_frontend.py
# Script para debuggear el frontend
import requests
import json

def test_frontend_api_calls():
    """Probar llamadas API desde el frontend."""
    print("=== DEBUGGEANDO FRONTEND API CALLS ===")
    
    # Simular llamada desde el frontend
    base_url = "http://localhost:8000"
    
    # Headers que usaría el frontend
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    try:
        # 1. Login
        print("1. Probando login...")
        login_data = {
            "email": "admin@example.com",
            "password": "admin123"
        }
        
        response = requests.post(
            f"{base_url}/auth/login", 
            json=login_data, 
            headers=headers
        )
        
        print(f"   Status: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            token_data = response.json()
            print(f"   ✅ Login exitoso")
            print(f"   Token: {token_data.get('access_token', 'No token')[:50]}...")
            
            # 2. Probar /auth/me con el token
            print("\n2. Probando /auth/me...")
            auth_headers = {
                **headers,
                "Authorization": f"Bearer {token_data['access_token']}"
            }
            
            me_response = requests.get(
                f"{base_url}/auth/me", 
                headers=auth_headers
            )
            
            print(f"   Status: {me_response.status_code}")
            print(f"   Headers: {dict(me_response.headers)}")
            
            if me_response.status_code == 200:
                user_data = me_response.json()
                print(f"   ✅ /auth/me exitoso")
                print(f"   User: {user_data}")
            else:
                print(f"   ❌ /auth/me falló: {me_response.text}")
                
        else:
            print(f"   ❌ Login falló: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_cors():
    """Probar CORS."""
    print("\n=== PROBANDO CORS ===")
    
    try:
        # Simular request desde el frontend
        headers = {
            "Origin": "http://localhost:3000",
            "Content-Type": "application/json",
        }
        
        response = requests.options(
            "http://localhost:8000/auth/login",
            headers=headers
        )
        
        print(f"OPTIONS Status: {response.status_code}")
        print(f"CORS Headers: {dict(response.headers)}")
        
    except Exception as e:
        print(f"❌ CORS Error: {e}")

if __name__ == "__main__":
    test_frontend_api_calls()
    test_cors()
