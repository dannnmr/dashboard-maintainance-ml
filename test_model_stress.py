#!/usr/bin/env python3
"""
Script de pruebas de estrés y validación cruzada para el modelo
de detección de anomalías en transformadores eléctricos.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Agregar paths necesarios
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'modelo'))

from backend.modelo.infer import load_artifacts, infer_from_last_24h
from backend.modelo.ensemble import best_thr_fbeta, smooth_alerts, metrics_auc

class ModelStressTester:
    """Clase para realizar pruebas de estrés del modelo."""
    
    def __init__(self):
        self.artifacts = None
        self.stress_results = {}
        
    def load_model(self):
        """Carga el modelo y sus artefactos."""
        try:
            self.artifacts = load_artifacts()
            print("✅ Modelo cargado para pruebas de estrés")
            return True
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Genera datos sintéticos para pruebas de estrés."""
        print(f"🔧 Generando {n_samples} muestras sintéticas...")
        
        ae, scaler_ae, feature_cols, meta, medians = self.artifacts
        
        # Generar datos basados en las medianas
        synthetic_data = pd.DataFrame(index=range(n_samples), columns=feature_cols)
        
        for col in feature_cols:
            if col in medians:
                # Generar datos alrededor de la mediana con variación
                base_value = medians[col]
                noise = np.random.normal(0, base_value * 0.1, n_samples)
                synthetic_data[col] = base_value + noise
            else:
                # Generar datos aleatorios si no hay mediana
                synthetic_data[col] = np.random.normal(0, 1, n_samples)
        
        return synthetic_data
    
    def test_edge_cases(self):
        """Prueba casos extremos del modelo."""
        print("\n🧪 PRUEBAS DE CASOS EXTREMOS")
        print("=" * 40)
        
        if not self.artifacts:
            print("❌ Modelo no cargado")
            return
            
        ae, scaler_ae, feature_cols, meta, medians = self.artifacts
        edge_cases = {}
        
        # Caso 1: Datos completamente normales
        print("📊 Caso 1: Datos completamente normales")
        normal_data = pd.DataFrame([medians] * meta['lookback'])
        try:
            result = infer_from_last_24h(normal_data)
            edge_cases['normal_data'] = result
            print(f"   ✅ Score: {result['score']:.4f}, Predicción: {result['pred']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            edge_cases['normal_data'] = {'error': str(e)}
        
        # Caso 2: Datos con valores extremos
        print("📊 Caso 2: Datos con valores extremos")
        extreme_data = pd.DataFrame([medians] * meta['lookback'])
        # Modificar algunos valores para que sean extremos
        extreme_data.iloc[-1, 0] = extreme_data.iloc[-1, 0] * 100  # Valor 100x mayor
        extreme_data.iloc[-2, 1] = extreme_data.iloc[-2, 1] * 0.01  # Valor muy pequeño
        
        try:
            result = infer_from_last_24h(extreme_data)
            edge_cases['extreme_data'] = result
            print(f"   ✅ Score: {result['score']:.4f}, Predicción: {result['pred']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            edge_cases['extreme_data'] = {'error': str(e)}
        
        # Caso 3: Datos con valores NaN
        print("📊 Caso 3: Datos con valores NaN")
        nan_data = pd.DataFrame([medians] * meta['lookback'])
        nan_data.iloc[-1, 0] = np.nan
        nan_data.iloc[-2, 1] = np.nan
        
        try:
            result = infer_from_last_24h(nan_data)
            edge_cases['nan_data'] = result
            print(f"   ✅ Score: {result['score']:.4f}, Predicción: {result['pred']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            edge_cases['nan_data'] = {'error': str(e)}
        
        # Caso 4: Datos con valores infinitos
        print("📊 Caso 4: Datos con valores infinitos")
        inf_data = pd.DataFrame([medians] * meta['lookback'])
        inf_data.iloc[-1, 0] = np.inf
        inf_data.iloc[-2, 1] = -np.inf
        
        try:
            result = infer_from_last_24h(inf_data)
            edge_cases['inf_data'] = result
            print(f"   ✅ Score: {result['score']:.4f}, Predicción: {result['pred']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            edge_cases['inf_data'] = {'error': str(e)}
        
        self.stress_results['edge_cases'] = edge_cases
    
    def test_performance_under_load(self):
        """Prueba el rendimiento del modelo bajo carga."""
        print("\n⚡ PRUEBAS DE RENDIMIENTO BAJO CARGA")
        print("=" * 40)
        
        if not self.artifacts:
            print("❌ Modelo no cargado")
            return
            
        ae, scaler_ae, feature_cols, meta, medians = self.artifacts
        
        # Generar datos sintéticos
        synthetic_data = self.generate_synthetic_data(10000)
        
        performance_results = {
            'batch_sizes': [1, 10, 50, 100, 500],
            'results': {}
        }
        
        for batch_size in performance_results['batch_sizes']:
            print(f"📊 Probando batch size: {batch_size}")
            
            processing_times = []
            predictions = []
            scores = []
            
            start_time = datetime.now()
            
            for i in range(0, len(synthetic_data) - meta['lookback'], batch_size):
                batch_start = datetime.now()
                
                for j in range(min(batch_size, len(synthetic_data) - i - meta['lookback'])):
                    idx = i + j
                    window_data = synthetic_data.iloc[idx:idx + meta['lookback']]
                    
                    try:
                        result = infer_from_last_24h(window_data)
                        predictions.append(result['pred'])
                        scores.append(result['score'])
                    except Exception as e:
                        predictions.append(0)
                        scores.append(0.0)
                
                batch_time = (datetime.now() - batch_start).total_seconds()
                processing_times.append(batch_time)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            performance_results['results'][batch_size] = {
                'total_time': total_time,
                'avg_batch_time': np.mean(processing_times),
                'predictions_per_second': len(predictions) / total_time,
                'total_predictions': len(predictions)
            }
            
            print(f"   ✅ Tiempo total: {total_time:.2f}s")
            print(f"   ✅ Predicciones/segundo: {len(predictions) / total_time:.2f}")
        
        self.stress_results['performance'] = performance_results
    
    def test_model_robustness(self):
        """Prueba la robustez del modelo con diferentes tipos de ruido."""
        print("\n🛡️  PRUEBAS DE ROBUSTEZ")
        print("=" * 40)
        
        if not self.artifacts:
            print("❌ Modelo no cargado")
            return
            
        ae, scaler_ae, feature_cols, meta, medians = self.artifacts
        
        # Generar datos base
        base_data = pd.DataFrame([medians] * meta['lookback'])
        
        robustness_results = {
            'noise_levels': [0.01, 0.05, 0.1, 0.2, 0.5],
            'results': {}
        }
        
        for noise_level in robustness_results['noise_levels']:
            print(f"📊 Probando nivel de ruido: {noise_level * 100}%")
            
            # Aplicar ruido gaussiano
            noisy_data = base_data.copy()
            for col in feature_cols:
                noise = np.random.normal(0, medians[col] * noise_level, meta['lookback'])
                noisy_data[col] = noisy_data[col] + noise
            
            try:
                result = infer_from_last_24h(noisy_data)
                robustness_results['results'][noise_level] = {
                    'score': result['score'],
                    'prediction': result['pred'],
                    'status': 'success'
                }
                print(f"   ✅ Score: {result['score']:.4f}, Predicción: {result['pred']}")
            except Exception as e:
                robustness_results['results'][noise_level] = {
                    'error': str(e),
                    'status': 'failed'
                }
                print(f"   ❌ Error: {e}")
        
        self.stress_results['robustness'] = robustness_results
    
    def test_memory_usage(self):
        """Prueba el uso de memoria del modelo."""
        print("\n💾 PRUEBAS DE USO DE MEMORIA")
        print("=" * 40)
        
        if not self.artifacts:
            print("❌ Modelo no cargado")
            return
            
        import psutil
        import os
        
        # Obtener uso de memoria inicial
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"📊 Memoria inicial: {initial_memory:.2f} MB")
        
        # Realizar múltiples predicciones
        ae, scaler_ae, feature_cols, meta, medians = self.artifacts
        synthetic_data = self.generate_synthetic_data(1000)
        
        memory_usage = [initial_memory]
        
        for i in range(0, len(synthetic_data) - meta['lookback'], 100):
            window_data = synthetic_data.iloc[i:i + meta['lookback']]
            
            try:
                result = infer_from_last_24h(window_data)
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_usage.append(current_memory)
                
                if i % 500 == 0:
                    print(f"   📊 Predicciones: {i}, Memoria: {current_memory:.2f} MB")
                    
            except Exception as e:
                print(f"   ❌ Error en predicción {i}: {e}")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        print(f"📊 Memoria final: {final_memory:.2f} MB")
        print(f"📊 Crecimiento de memoria: {memory_growth:.2f} MB")
        
        self.stress_results['memory_usage'] = {
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'memory_growth': memory_growth,
            'memory_usage_sequence': memory_usage
        }
    
    def generate_stress_report(self):
        """Genera un reporte de las pruebas de estrés."""
        print("\n📋 GENERANDO REPORTE DE PRUEBAS DE ESTRÉS")
        print("=" * 50)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'stress_tests': self.stress_results,
            'summary': {}
        }
        
        # Resumen de casos extremos
        if 'edge_cases' in self.stress_results:
            edge_success = sum(1 for case in self.stress_results['edge_cases'].values() 
                             if 'error' not in case)
            report['summary']['edge_cases_success_rate'] = edge_success / len(self.stress_results['edge_cases'])
        
        # Resumen de rendimiento
        if 'performance' in self.stress_results:
            best_throughput = max(
                result['predictions_per_second'] 
                for result in self.stress_results['performance']['results'].values()
            )
            report['summary']['max_throughput'] = best_throughput
        
        # Resumen de robustez
        if 'robustness' in self.stress_results:
            robustness_success = sum(1 for result in self.stress_results['robustness']['results'].values() 
                                   if result['status'] == 'success')
            report['summary']['robustness_success_rate'] = robustness_success / len(self.stress_results['robustness']['results'])
        
        # Guardar reporte
        report_path = 'model_stress_test_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Reporte de estrés guardado en: {report_path}")
        
        # Mostrar resumen
        print("\n📊 RESUMEN DE PRUEBAS DE ESTRÉS")
        print("=" * 40)
        
        if 'edge_cases_success_rate' in report['summary']:
            print(f"✅ Casos extremos exitosos: {report['summary']['edge_cases_success_rate']*100:.1f}%")
        
        if 'max_throughput' in report['summary']:
            print(f"✅ Máximo throughput: {report['summary']['max_throughput']:.2f} pred/seg")
        
        if 'robustness_success_rate' in report['summary']:
            print(f"✅ Robustez exitosa: {report['summary']['robustness_success_rate']*100:.1f}%")
        
        if 'memory_usage' in self.stress_results:
            memory_info = self.stress_results['memory_usage']
            print(f"✅ Crecimiento de memoria: {memory_info['memory_growth']:.2f} MB")
        
        return report
    
    def run_stress_tests(self):
        """Ejecuta todas las pruebas de estrés."""
        print("🚀 INICIANDO PRUEBAS DE ESTRÉS DEL MODELO")
        print("=" * 50)
        
        if not self.load_model():
            return False
        
        self.test_edge_cases()
        self.test_performance_under_load()
        self.test_model_robustness()
        self.test_memory_usage()
        
        report = self.generate_stress_report()
        
        print("\n✅ PRUEBAS DE ESTRÉS COMPLETADAS")
        print("=" * 50)
        
        return True

def main():
    """Función principal para ejecutar las pruebas de estrés."""
    tester = ModelStressTester()
    success = tester.run_stress_tests()
    
    if success:
        print("\n🎉 Todas las pruebas de estrés se completaron exitosamente.")
        print("📊 Revisa el archivo generado:")
        print("   • model_stress_test_report.json - Reporte de pruebas de estrés")
    else:
        print("\n❌ Las pruebas de estrés fallaron. Revisa los errores anteriores.")

if __name__ == "__main__":
    main()

