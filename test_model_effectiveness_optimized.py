#!/usr/bin/env python3
"""
Script optimizado de validación de efectividad del modelo de detección de anomalías
para transformadores eléctricos.

Versión optimizada para evitar problemas de rendimiento con TensorFlow.
"""

import sys
import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configurar TensorFlow para evitar warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Agregar el directorio del backend al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'modelo'))

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    from backend.modelo.infer import load_artifacts, infer_from_last_24h
    from backend.modelo.ensemble import best_thr_fbeta, smooth_alerts, metrics_auc
    from sklearn.metrics import (
        classification_report, confusion_matrix, 
        roc_curve, precision_recall_curve, roc_auc_score
    )
except ImportError as e:
    print(f"❌ Error importando dependencias: {e}")
    sys.exit(1)

class OptimizedModelTester:
    """Clase optimizada para realizar pruebas de efectividad del modelo."""
    
    def __init__(self):
        self.results = {}
        self.artifacts = None
        self.data_paths = {
            'train': 'data/capa_gold/features_transformador/transformer_features_train_20250928_1834.parquet',
            'validation': 'data/capa_gold/features_transformador/transformer_features_validation_20250928_1834.parquet',
            'complete': 'data/capa_gold/features_transformador/transformer_features_complete_20250928_1834.parquet'
        }
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Carga los datasets de entrenamiento, validación y completo."""
        print("📊 Cargando datasets...")
        datasets = {}
        
        for name, path in self.data_paths.items():
            if os.path.exists(path):
                datasets[name] = pd.read_parquet(path)
                print(f"✅ {name}: {len(datasets[name])} registros cargados")
            else:
                print(f"⚠️  Archivo no encontrado: {path}")
                
        return datasets
    
    def load_model_artifacts(self):
        """Carga los artefactos del modelo entrenado."""
        print("🤖 Cargando artefactos del modelo...")
        try:
            self.artifacts = load_artifacts()
            print("✅ Artefactos del modelo cargados correctamente")
            return True
        except Exception as e:
            print(f"❌ Error cargando artefactos: {e}")
            return False
    
    def analyze_model_metadata(self):
        """Analiza los metadatos del modelo."""
        print("\n📋 ANÁLISIS DE METADATOS DEL MODELO")
        print("=" * 50)
        
        if not self.artifacts:
            print("❌ No se pudieron cargar los artefactos")
            return
            
        ae, scaler_ae, feature_cols, meta, medians = self.artifacts
        
        print(f"🔧 Configuración del modelo:")
        print(f"   • Lookback: {meta['lookback']} horas")
        print(f"   • Horizon shift: {meta['horizon_shift']} horas")
        print(f"   • Clases: {meta['classes']}")
        print(f"   • Features utilizadas: {len(feature_cols)}")
        
        print(f"\n📈 Métricas de performance:")
        print(f"   • ROC AUC (AE): {meta['ae_roc_auc']:.4f}")
        print(f"   • PR AUC (AE): {meta['ae_pr_auc']:.4f}")
        print(f"   • ROC AUC (IF): {meta['if_roc_auc']:.4f}")
        print(f"   • PR AUC (IF): {meta['if_pr_auc']:.4f}")
        print(f"   • ROC AUC (Ensemble): {meta['ens_roc_auc']:.4f}")
        print(f"   • PR AUC (Ensemble): {meta['ens_pr_auc']:.4f}")
        
        print(f"\n⚙️  Configuración operacional:")
        print(f"   • Usar solo AE: {meta['operate_with_ae_only']}")
        print(f"   • Threshold operacional: {meta['operate_thr']:.4f}")
        print(f"   • Smoothing K: {meta['smoothing_k']}")
        print(f"   • Smoothing M: {meta['smoothing_m']}")
        
        self.results['metadata'] = meta
    
    def test_historical_data_optimized(self, datasets: Dict[str, pd.DataFrame]):
        """Prueba el modelo con datos históricos de forma optimizada."""
        print("\n🔍 PRUEBAS CON DATOS HISTÓRICOS (OPTIMIZADO)")
        print("=" * 50)
        
        if not self.artifacts:
            print("❌ No se pudieron cargar los artefactos")
            return
            
        ae, scaler_ae, feature_cols, meta, medians = self.artifacts
        historical_results = {}
        
        for dataset_name, df in datasets.items():
            print(f"\n📊 Procesando dataset: {dataset_name}")
            
            if 'estado_futuro' not in df.columns:
                print(f"⚠️  Dataset {dataset_name} no tiene columna 'estado_futuro'")
                continue
            
            # Limitar el número de muestras para evitar timeouts
            max_samples = min(len(df), 1000)  # Procesar máximo 1000 muestras
            df_sample = df.head(max_samples)
            
            # Filtrar solo las columnas de features
            df_features = df_sample[feature_cols].copy()
            
            # Manejar valores faltantes e infinitos
            df_features = df_features.replace([np.inf, -np.inf], np.nan)
            df_features = df_features.fillna(medians)
            
            # Generar secuencias para predicción
            predictions = []
            scores = []
            
            print(f"   • Procesando {max_samples} registros (muestra)...")
            
            # Procesar en lotes más pequeños
            batch_size = 50
            for i in range(meta['lookback'], max_samples, batch_size):
                batch_end = min(i + batch_size, max_samples)
                
                for j in range(i, batch_end):
                    window_data = df_features.iloc[j-meta['lookback']:j]
                    
                    try:
                        result = infer_from_last_24h(window_data)
                        predictions.append(result['pred'])
                        scores.append(result['score'])
                    except Exception as e:
                        predictions.append(0)
                        scores.append(0.0)
                        
                # Mostrar progreso
                progress = (batch_end / max_samples) * 100
                print(f"   📊 Progreso: {progress:.1f}%")
                    
            # Completar con ceros al inicio
            predictions = [0] * meta['lookback'] + predictions
            scores = [0.0] * meta['lookback'] + scores
            
            # Obtener etiquetas reales
            y_true = (df_sample['estado_futuro'] != 'NORMAL').astype(int).values
            
            # Calcular métricas
            if len(y_true) == len(predictions):
                roc_auc, pr_auc = metrics_auc(y_true, np.array(scores))
                
                historical_results[dataset_name] = {
                    'predictions': predictions,
                    'scores': scores,
                    'y_true': y_true.tolist(),
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc,
                    'n_samples': len(df_sample)
                }
                
                print(f"   ✅ ROC AUC: {roc_auc:.4f}")
                print(f"   ✅ PR AUC: {pr_auc:.4f}")
                print(f"   ✅ Anomalías detectadas: {sum(predictions)}")
                print(f"   ✅ Anomalías reales: {sum(y_true)}")
        
        self.results['historical'] = historical_results
    
    def simulate_realtime_predictions_optimized(self, datasets: Dict[str, pd.DataFrame]):
        """Simula predicciones en tiempo real de forma optimizada."""
        print("\n⏱️  SIMULACIÓN DE PREDICCIONES EN TIEMPO REAL (OPTIMIZADO)")
        print("=" * 50)
        
        if not self.artifacts:
            print("❌ No se pudieron cargar los artefactos")
            return
            
        ae, scaler_ae, feature_cols, meta, medians = self.artifacts
        
        # Usar el dataset completo para simulación
        if 'complete' not in datasets:
            print("❌ No se encontró dataset completo para simulación")
            return
            
        df = datasets['complete']
        df_features = df[feature_cols].copy()
        df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(medians)
        
        # Limitar muestras para simulación más rápida
        max_samples = min(len(df_features), 500)
        df_sample = df_features.head(max_samples)
        
        print(f"📊 Simulando predicciones en tiempo real con {max_samples} registros...")
        
        realtime_results = {
            'timestamps': [],
            'predictions': [],
            'scores': [],
            'processing_times': [],
            'alerts_triggered': []
        }
        
        # Simular procesamiento cada 24 horas
        start_time = time.time()
        alert_count = 0
        
        step_size = 24  # Cada 24 horas
        for i in range(meta['lookback'], max_samples, step_size):
            window_data = df_sample.iloc[i-meta['lookback']:i]
            
            # Medir tiempo de procesamiento
            pred_start = time.time()
            
            try:
                result = infer_from_last_24h(window_data)
                processing_time = time.time() - pred_start
                
                realtime_results['timestamps'].append(i)
                realtime_results['predictions'].append(result['pred'])
                realtime_results['scores'].append(result['score'])
                realtime_results['processing_times'].append(processing_time)
                
                if result['pred'] == 1:
                    alert_count += 1
                    realtime_results['alerts_triggered'].append({
                        'timestamp': i,
                        'score': result['score'],
                        'threshold': result['operate_thr']
                    })
                    
            except Exception as e:
                realtime_results['timestamps'].append(i)
                realtime_results['predictions'].append(0)
                realtime_results['scores'].append(0.0)
                realtime_results['processing_times'].append(0.0)
            
            # Mostrar progreso
            progress = (i / max_samples) * 100
            print(f"   📊 Progreso: {progress:.1f}%")
        
        total_time = time.time() - start_time
        
        print(f"✅ Simulación completada:")
        print(f"   • Tiempo total: {total_time:.2f} segundos")
        print(f"   • Predicciones realizadas: {len(realtime_results['predictions'])}")
        print(f"   • Alertas generadas: {alert_count}")
        print(f"   • Tiempo promedio por predicción: {np.mean(realtime_results['processing_times']):.4f} segundos")
        
        self.results['realtime'] = realtime_results
    
    def generate_performance_metrics(self):
        """Genera métricas detalladas de performance."""
        print("\n📊 MÉTRICAS DETALLADAS DE PERFORMANCE")
        print("=" * 50)
        
        metrics_summary = {}
        
        # Métricas históricas
        if 'historical' in self.results:
            print("\n📈 Métricas Históricas:")
            for dataset_name, results in self.results['historical'].items():
                y_true = np.array(results['y_true'])
                predictions = np.array(results['predictions'])
                scores = np.array(results['scores'])
                
                # Métricas básicas
                tp = np.sum((y_true == 1) & (predictions == 1))
                fp = np.sum((y_true == 0) & (predictions == 1))
                fn = np.sum((y_true == 1) & (predictions == 0))
                tn = np.sum((y_true == 0) & (predictions == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics_summary[dataset_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': results['roc_auc'],
                    'pr_auc': results['pr_auc'],
                    'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
                }
                
                print(f"\n   {dataset_name.upper()}:")
                print(f"   • Precisión: {precision:.4f}")
                print(f"   • Recall: {recall:.4f}")
                print(f"   • F1-Score: {f1:.4f}")
                print(f"   • ROC AUC: {results['roc_auc']:.4f}")
                print(f"   • PR AUC: {results['pr_auc']:.4f}")
        
        # Métricas tiempo real
        if 'realtime' in self.results:
            realtime = self.results['realtime']
            print(f"\n⏱️  Métricas Tiempo Real:")
            print(f"   • Predicciones procesadas: {len(realtime['predictions'])}")
            print(f"   • Alertas generadas: {len(realtime['alerts_triggered'])}")
            print(f"   • Tiempo promedio por predicción: {np.mean(realtime['processing_times']):.4f}s")
            print(f"   • Tiempo máximo: {np.max(realtime['processing_times']):.4f}s")
            print(f"   • Tiempo mínimo: {np.min(realtime['processing_times']):.4f}s")
        
        self.results['performance_metrics'] = metrics_summary
    
    def create_visualizations(self):
        """Crea visualizaciones de los resultados."""
        print("\n📊 GENERANDO VISUALIZACIONES")
        print("=" * 50)
        
        try:
            # Configurar estilo
            plt.style.use('default')
            fig = plt.figure(figsize=(20, 15))
            
            # 1. Matriz de confusión
            if 'historical' in self.results:
                ax1 = plt.subplot(2, 3, 1)
                for dataset_name, results in self.results['historical'].items():
                    y_true = np.array(results['y_true'])
                    predictions = np.array(results['predictions'])
                    
                    cm = confusion_matrix(y_true, predictions)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['Normal', 'Anómalo'],
                               yticklabels=['Normal', 'Anómalo'])
                    plt.title(f'Matriz de Confusión - {dataset_name.title()}')
                    plt.ylabel('Etiqueta Real')
                    plt.xlabel('Predicción')
                    break  # Solo mostrar una matriz
            
            # 2. Curva ROC
            ax2 = plt.subplot(2, 3, 2)
            if 'historical' in self.results:
                for dataset_name, results in self.results['historical'].items():
                    y_true = np.array(results['y_true'])
                    scores = np.array(results['scores'])
                    
                    fpr, tpr, _ = roc_curve(y_true, scores)
                    auc_score = roc_auc_score(y_true, scores)
                    
                    plt.plot(fpr, tpr, label=f'{dataset_name.title()} (AUC = {auc_score:.3f})')
                
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('Tasa de Falsos Positivos')
                plt.ylabel('Tasa de Verdaderos Positivos')
                plt.title('Curva ROC')
                plt.legend()
                plt.grid(True)
            
            # 3. Curva Precision-Recall
            ax3 = plt.subplot(2, 3, 3)
            if 'historical' in self.results:
                for dataset_name, results in self.results['historical'].items():
                    y_true = np.array(results['y_true'])
                    scores = np.array(results['scores'])
                    
                    precision, recall, _ = precision_recall_curve(y_true, scores)
                    pr_auc = results['pr_auc']
                    
                    plt.plot(recall, precision, label=f'{dataset_name.title()} (PR-AUC = {pr_auc:.3f})')
                
                plt.xlabel('Recall')
                plt.ylabel('Precisión')
                plt.title('Curva Precision-Recall')
                plt.legend()
                plt.grid(True)
            
            # 4. Distribución de scores
            ax4 = plt.subplot(2, 3, 4)
            if 'historical' in self.results:
                for dataset_name, results in self.results['historical'].items():
                    scores = np.array(results['scores'])
                    plt.hist(scores, bins=30, alpha=0.7, label=dataset_name.title())
                
                # Marcar threshold
                if 'metadata' in self.results:
                    threshold = self.results['metadata']['operate_thr']
                    plt.axvline(threshold, color='red', linestyle='--', 
                               label=f'Threshold = {threshold:.3f}')
                
                plt.xlabel('Score de Anomalía')
                plt.ylabel('Frecuencia')
                plt.title('Distribución de Scores')
                plt.legend()
                plt.grid(True)
            
            # 5. Tiempo de procesamiento en tiempo real
            ax5 = plt.subplot(2, 3, 5)
            if 'realtime' in self.results:
                processing_times = self.results['realtime']['processing_times']
                plt.plot(processing_times, marker='o', markersize=3)
                plt.xlabel('Número de Predicción')
                plt.ylabel('Tiempo (segundos)')
                plt.title('Tiempo de Procesamiento en Tiempo Real')
                plt.grid(True)
            
            # 6. Alertas en el tiempo
            ax6 = plt.subplot(2, 3, 6)
            if 'realtime' in self.results:
                predictions = self.results['realtime']['predictions']
                timestamps = self.results['realtime']['timestamps']
                
                plt.plot(timestamps, predictions, marker='o', markersize=4)
                plt.xlabel('Timestamp')
                plt.ylabel('Predicción (0=Normal, 1=Anómalo)')
                plt.title('Alertas en Tiempo Real')
                plt.grid(True)
            
            plt.tight_layout()
            
            # Guardar visualización
            output_path = 'model_effectiveness_analysis_optimized.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualizaciones guardadas en: {output_path}")
            
            plt.close()  # Cerrar la figura para liberar memoria
            
        except Exception as e:
            print(f"⚠️  Error generando visualizaciones: {e}")
    
    def generate_report(self):
        """Genera un reporte completo de la efectividad del modelo."""
        print("\n📋 GENERANDO REPORTE DE EFECTIVIDAD")
        print("=" * 50)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': self.results.get('metadata', {}),
            'historical_performance': self.results.get('historical', {}),
            'realtime_performance': self.results.get('realtime', {}),
            'performance_metrics': self.results.get('performance_metrics', {}),
            'recommendations': []
        }
        
        # Generar recomendaciones
        recommendations = []
        
        if 'metadata' in self.results:
            meta = self.results['metadata']
            
            if meta['ae_roc_auc'] < 0.7:
                recommendations.append("⚠️  El ROC AUC del Autoencoder está por debajo de 0.7. Considerar retrenar el modelo.")
            
            if meta['ae_pr_auc'] < 0.1:
                recommendations.append("⚠️  El PR AUC del Autoencoder es muy bajo. El modelo puede tener dificultades con datos desbalanceados.")
            
            if meta['operate_thr'] > 0.8:
                recommendations.append("⚠️  El threshold operacional es muy alto. Esto puede resultar en pocas detecciones.")
            
        if 'realtime' in self.results:
            avg_time = np.mean(self.results['realtime']['processing_times'])
            if avg_time > 1.0:
                recommendations.append("⚠️  El tiempo promedio de procesamiento es alto (>1s). Considerar optimización.")
        
        report['recommendations'] = recommendations
        
        # Guardar reporte
        report_path = 'model_effectiveness_report_optimized.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Reporte guardado en: {report_path}")
        
        # Mostrar resumen
        print("\n📊 RESUMEN DE EFECTIVIDAD DEL MODELO")
        print("=" * 50)
        
        if 'metadata' in self.results:
            meta = self.results['metadata']
            print(f"🎯 Performance General:")
            print(f"   • ROC AUC (AE): {meta['ae_roc_auc']:.4f}")
            print(f"   • PR AUC (AE): {meta['ae_pr_auc']:.4f}")
            print(f"   • Threshold Operacional: {meta['operate_thr']:.4f}")
        
        if 'performance_metrics' in self.results:
            for dataset_name, metrics in self.results['performance_metrics'].items():
                print(f"\n📈 {dataset_name.upper()}:")
                print(f"   • Precisión: {metrics['precision']:.4f}")
                print(f"   • Recall: {metrics['recall']:.4f}")
                print(f"   • F1-Score: {metrics['f1_score']:.4f}")
        
        if recommendations:
            print(f"\n💡 RECOMENDACIONES:")
            for rec in recommendations:
                print(f"   {rec}")
        
        return report
    
    def run_complete_test(self):
        """Ejecuta todas las pruebas de efectividad de forma optimizada."""
        print("🚀 INICIANDO PRUEBAS DE EFECTIVIDAD DEL MODELO (OPTIMIZADO)")
        print("=" * 60)
        
        # Cargar datos y modelo
        datasets = self.load_data()
        if not self.load_model_artifacts():
            return False
        
        # Ejecutar pruebas
        self.analyze_model_metadata()
        self.test_historical_data_optimized(datasets)
        self.simulate_realtime_predictions_optimized(datasets)
        self.generate_performance_metrics()
        self.create_visualizations()
        
        # Generar reporte final
        report = self.generate_report()
        
        print("\n✅ PRUEBAS COMPLETADAS EXITOSAMENTE")
        print("=" * 60)
        
        return True

def main():
    """Función principal para ejecutar las pruebas optimizadas."""
    tester = OptimizedModelTester()
    success = tester.run_complete_test()
    
    if success:
        print("\n🎉 Todas las pruebas se completaron exitosamente.")
        print("📊 Revisa los archivos generados:")
        print("   • model_effectiveness_analysis_optimized.png - Visualizaciones")
        print("   • model_effectiveness_report_optimized.json - Reporte detallado")
    else:
        print("\n❌ Las pruebas fallaron. Revisa los errores anteriores.")

if __name__ == "__main__":
    main()

