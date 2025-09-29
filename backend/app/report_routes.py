from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from typing import Optional, List
import pandas as pd
import io
import os
from datetime import datetime, timedelta
from pathlib import Path
import json

router = APIRouter(tags=["reports"])

@router.get("/reports/historical-data/csv")
async def export_historical_data_csv(
    transformer_id: str = Query("TR01", description="ID del transformador"),
    start_date: Optional[str] = Query(None, description="Fecha inicio (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Fecha fin (YYYY-MM-DD)"),
    sample_hours: int = Query(1, description="Intervalo de muestreo en horas")
):
    """
    Exportar datos históricos del transformador a CSV
    """
    try:
        # Cargar datos históricos
        data_path = Path(__file__).parent.parent.parent / "data" / "capa_gold" / "features_transformador" / "features_complete"
        
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Ruta de datos no encontrada")
        
        # Obtener archivos disponibles
        available_months = []
        for year_dir in data_path.iterdir():
            if year_dir.is_dir() and year_dir.name.startswith("year="):
                year = int(year_dir.name.split("=")[1])
                for month_dir in year_dir.iterdir():
                    if month_dir.is_dir() and month_dir.name.startswith("month="):
                        month = int(month_dir.name.split("=")[1])
                        parquet_files = list(month_dir.glob("*.parquet"))
                        if parquet_files:
                            available_months.append({
                                "year": year,
                                "month": month,
                                "file": parquet_files[0]
                            })
        
        # Cargar datos
        all_data = []
        for month_info in available_months:
            try:
                df = pd.read_parquet(month_info["file"])
                df["year"] = month_info["year"]
                df["month"] = month_info["month"]
                all_data.append(df)
            except Exception as e:
                print(f"Error cargando mes {month_info['year']}-{month_info['month']}: {e}")
                continue
        
        if not all_data:
            raise HTTPException(status_code=404, detail="No se encontraron datos históricos")
        
        # Combinar datos
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values("timestamp")
        
        # Filtrar por fechas si se proporcionan
        if start_date:
            start_dt = pd.to_datetime(start_date)
            combined_df = combined_df[combined_df["timestamp"] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            combined_df = combined_df[combined_df["timestamp"] <= end_dt]
        
        # Muestrear datos
        if sample_hours > 1:
            combined_df = combined_df.iloc[::sample_hours]
        
        # Seleccionar columnas relevantes para el reporte
        report_columns = [
            "timestamp",
            "temp_oil_value",
            "temp_ambient_value", 
            "voltage_value",
            "current_load_value",
            "power_apparent_value",
            "tap_position_value",
            "estado_operacional",
            "nivel_severidad",
            "temp_spot_hot_value",
            "gradient_hot_oil"
        ]
        
        # Filtrar columnas que existen
        existing_columns = [col for col in report_columns if col in combined_df.columns]
        report_df = combined_df[existing_columns].copy()
        
        # Renombrar columnas para el reporte
        column_mapping = {
            "timestamp": "Fecha_Hora",
            "temp_oil_value": "Temp_Aceite_C",
            "temp_ambient_value": "Temp_Ambiente_C", 
            "voltage_value": "Voltaje_V",
            "current_load_value": "Corriente_A",
            "power_apparent_value": "Potencia_kVA",
            "tap_position_value": "Posicion_Tap",
            "estado_operacional": "Estado_Operacional",
            "nivel_severidad": "Nivel_Severidad",
            "temp_spot_hot_value": "Temp_Punto_Caliente_C",
            "gradient_hot_oil": "Gradiente_Caliente_Aceite"
        }
        
        report_df = report_df.rename(columns=column_mapping)
        
        # Crear CSV en memoria
        output = io.StringIO()
        report_df.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        
        # Preparar respuesta
        csv_content = output.getvalue()
        output.close()
        
        # Generar nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"datos_historicos_{transformer_id}_{timestamp}.csv"
        
        return StreamingResponse(
            io.BytesIO(csv_content.encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        print(f"❌ Error generando CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error generando reporte CSV: {str(e)}")

@router.get("/reports/historical-data/pdf")
async def generate_historical_data_pdf(
    transformer_id: str = Query("TR01", description="ID del transformador"),
    start_date: Optional[str] = Query(None, description="Fecha inicio (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Fecha fin (YYYY-MM-DD)"),
    sample_hours: int = Query(6, description="Intervalo de muestreo en horas")
):
    """
    Generar reporte PDF con datos históricos del transformador
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
        # Cargar datos históricos (mismo código que CSV)
        data_path = Path(__file__).parent.parent.parent / "data" / "capa_gold" / "features_transformador" / "features_complete"
        
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Ruta de datos no encontrada")
        
        # Obtener y cargar datos
        available_months = []
        for year_dir in data_path.iterdir():
            if year_dir.is_dir() and year_dir.name.startswith("year="):
                year = int(year_dir.name.split("=")[1])
                for month_dir in year_dir.iterdir():
                    if month_dir.is_dir() and month_dir.name.startswith("month="):
                        month = int(month_dir.name.split("=")[1])
                        parquet_files = list(month_dir.glob("*.parquet"))
                        if parquet_files:
                            available_months.append({
                                "year": year,
                                "month": month,
                                "file": parquet_files[0]
                            })
        
        all_data = []
        for month_info in available_months:
            try:
                df = pd.read_parquet(month_info["file"])
                df["year"] = month_info["year"]
                df["month"] = month_info["month"]
                all_data.append(df)
            except Exception as e:
                print(f"Error cargando mes {month_info['year']}-{month_info['month']}: {e}")
                continue
        
        if not all_data:
            raise HTTPException(status_code=404, detail="No se encontraron datos históricos")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values("timestamp")
        
        # Filtrar por fechas
        if start_date:
            start_dt = pd.to_datetime(start_date)
            combined_df = combined_df[combined_df["timestamp"] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            combined_df = combined_df[combined_df["timestamp"] <= end_dt]
        
        # Muestrear datos
        if sample_hours > 1:
            combined_df = combined_df.iloc[::sample_hours]
        
        # Crear PDF en memoria
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Estilos
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # Contenido del PDF
        story = []
        
        # Título
        story.append(Paragraph("Reporte de Datos Históricos", title_style))
        story.append(Paragraph(f"Transformador: {transformer_id}", heading_style))
        story.append(Spacer(1, 12))
        
        # Información del reporte
        report_info = [
            ["Parámetro", "Valor"],
            ["Transformador", transformer_id],
            ["Fecha de Generación", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Período", f"{start_date or 'Inicio'} - {end_date or 'Actual'}"],
            ["Total de Registros", str(len(combined_df))],
            ["Intervalo de Muestreo", f"{sample_hours} horas"]
        ]
        
        info_table = Table(report_info, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(info_table)
        story.append(Spacer(1, 20))
        
        # Estadísticas de los datos
        story.append(Paragraph("Estadísticas de los Datos", heading_style))
        
        # Seleccionar columnas numéricas para estadísticas
        numeric_columns = [
            "temp_oil_value", "temp_ambient_value", "voltage_value", 
            "current_load_value", "power_apparent_value", "temp_spot_hot_value"
        ]
        
        existing_numeric = [col for col in numeric_columns if col in combined_df.columns]
        
        if existing_numeric:
            stats_data = [["Métrica", "Promedio", "Mínimo", "Máximo", "Desv. Est."]]
            
            for col in existing_numeric:
                col_name = col.replace("_value", "").replace("_", " ").title()
                stats = combined_df[col].describe()
                stats_data.append([
                    col_name,
                    f"{stats['mean']:.2f}",
                    f"{stats['min']:.2f}",
                    f"{stats['max']:.2f}",
                    f"{stats['std']:.2f}"
                ])
            
            stats_table = Table(stats_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(stats_table)
        
        story.append(Spacer(1, 20))
        
        # Muestra de datos (últimos 20 registros)
        story.append(Paragraph("Muestra de Datos (Últimos 20 Registros)", heading_style))
        
        sample_columns = [
            "timestamp", "temp_oil_value", "voltage_value", 
            "current_load_value", "power_apparent_value", "temp_spot_hot_value"
        ]
        
        existing_sample = [col for col in sample_columns if col in combined_df.columns]
        sample_df = combined_df[existing_sample].tail(20).copy()
        
        # Formatear datos para la tabla
        table_data = []
        
        # Encabezados
        headers = []
        for col in existing_sample:
            if col == "timestamp":
                headers.append("Fecha/Hora")
            else:
                headers.append(col.replace("_value", "").replace("_", " ").title())
        table_data.append(headers)
        
        # Datos
        for _, row in sample_df.iterrows():
            row_data = []
            for col in existing_sample:
                if col == "timestamp":
                    row_data.append(str(row[col])[:19])  # Truncar timestamp
                else:
                    row_data.append(f"{row[col]:.2f}")
            table_data.append(row_data)
        
        data_table = Table(table_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
        data_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(data_table)
        
        # Construir PDF
        doc.build(story)
        buffer.seek(0)
        
        # Preparar respuesta
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reporte_historico_{transformer_id}_{timestamp}.pdf"
        
        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        print(f"❌ Error generando PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error generando reporte PDF: {str(e)}")

@router.get("/reports/predictions/csv")
async def export_predictions_csv(
    start_date: Optional[str] = Query(None, description="Fecha inicio (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Fecha fin (YYYY-MM-DD)")
):
    """
    Exportar predicciones históricas a CSV
    """
    try:
        # Por ahora usamos datos simulados, en producción vendría de la base de datos
        predictions_data = []
        
        # Generar predicciones de ejemplo
        for i in range(20):
            predictions_data.append({
                "fecha_prediccion": (datetime.now() - timedelta(hours=i*12)).strftime("%Y-%m-%d %H:%M:%S"),
                "score_anomalia": round(0.1 + (i * 0.03), 4),
                "estado_predicho": "NORMAL" if i < 15 else "CRITICO",
                "horizonte_horas": 360,
                "confidence": round(0.85 + (i * 0.005), 3),
                "modelo_version": "v1.0"
            })
        
        # Crear DataFrame
        df = pd.DataFrame(predictions_data)
        
        # Crear CSV en memoria
        output = io.StringIO()
        df.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        
        # Preparar respuesta
        csv_content = output.getvalue()
        output.close()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predicciones_{timestamp}.csv"
        
        return StreamingResponse(
            io.BytesIO(csv_content.encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        print(f"❌ Error generando CSV de predicciones: {e}")
        raise HTTPException(status_code=500, detail=f"Error generando CSV de predicciones: {str(e)}")

@router.get("/reports/available-transformers")
async def get_available_transformers():
    """
    Obtener lista de transformadores disponibles
    """
    try:
        # Por ahora retornamos transformadores hardcodeados
        # En producción esto vendría de la base de datos o archivos
        transformers = [
            {"id": "TR01", "name": "Transformador Principal", "location": "Subestación Central"},
            {"id": "TR02", "name": "Transformador Secundario", "location": "Subestación Norte"},
            {"id": "TR03", "name": "Transformador Auxiliar", "location": "Subestación Sur"}
        ]
        
        return {"transformers": transformers}
        
    except Exception as e:
        print(f"❌ Error obteniendo transformadores: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo transformadores: {str(e)}")
