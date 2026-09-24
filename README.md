<div align="center">

# Laboratorio 7 · Spark MLlib

### Perfiles laborales y predicción salarial con ENEIC

**CC3084 · Data Science · Sección 10 · Grupo 1 · Segundo semestre 2026**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.1-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Estado](https://img.shields.io/badge/avance-75%25-2563EB)](#estado-del-avance)

</div>

## Propósito

Este repositorio contiene un flujo reproducible en PySpark para armonizar las bases de Personas de la ENEIC, describir la población asalariada elegible, identificar perfiles mediante KMeans y comparar modelos de regresión lineal y Random Forest para estimar el salario mensual.

El avance mantiene 2026T1 completamente reservado como prueba final. Los resultados son no ponderados y describen los registros analizados; no son estimaciones oficiales de la población guatemalteca ni implican relaciones causales.

## Integrantes

| Integrante | Carné |
|---|---:|
| Jorge Gabriel Palacios Sales | 231385 |
| Pablo Daniel Barillas Moreno | 22193 |
| Roberto Emiliano Otoniel | 23968 |

## Estado del avance

| Actividad de la guía | Estado |
|---|---|
| 1. Carga, armonización y calidad | Completa |
| 2. Estadística descriptiva | Completa |
| 3. Correlaciones con MLlib | Completa |
| 4. Segmentación KMeans | Completa |
| 5. Pipeline de regresión lineal | Completa |
| 6. Pipeline de Random Forest | Completa |
| 7. Reentrenamiento 2025 y prueba 2026 | Preparada; reservada para la entrega final |
| 8. Visualización y análisis de errores | Pendiente para la entrega final |

El avance oficial solicita las actividades 1–4. Este repositorio también deja ejecutadas las actividades 5–6 y preparados los datos de prueba.

## Estructura

```text
Lab-7/
├── notebooks/
│   └── Laboratorio_7_Spark_MLlib_Avance.ipynb
├── src/lab7/
│   ├── config.py
│   ├── data.py
│   ├── analysis.py
│   ├── modeling.py
│   └── visualization.py
├── outputs/
│   ├── figures/
│   └── tables/
├── reports/
│   ├── informe_avance.tex
│   └── informe_avance.pdf
├── ficha_repositorio/
│   ├── Ficha_Repositorio_Laboratorio_7.tex
│   └── Ficha_Repositorio_Laboratorio_7.pdf
├── scripts/
│   ├── build_notebook.py
│   └── build_report.py
├── tests/
└── requirements.txt
```

Los Excel, Parquet y modelos entrenados se mantienen fuera de Git en `working_dir/eneic/` por tamaño. El notebook conserva las rutas configurables mediante variables de entorno.

## Datos requeridos

Descargar de la [página oficial de ENEIC del INE](https://www.ine.gob.gt/encuesta-nacional-de-empleo-e-ingresos/) únicamente las bases de **Personas** y sus diccionarios para:

- 2025: trimestres I, II, III y IV.
- 2026: trimestre I.

Ubicación esperada dentro del contenedor:

```text
/opt/app/working_dir/eneic/raw
```

## Ejecución

Desde la raíz del proyecto que contiene `docker-compose.yml`:

```bash
docker compose up -d
docker ps
```

Abrir `http://localhost:8888`, ingresar a `notebooks/Lab-7/notebooks/` y ejecutar la libreta completa. También puede ejecutarse sin interfaz:

```bash
docker exec pyspark311-jdk17 bash -lc \
  "cd /opt/app/notebooks/Lab-7 && jupyter nbconvert \
  --to notebook --execute notebooks/Laboratorio_7_Spark_MLlib_Avance.ipynb \
  --output Laboratorio_7_Spark_MLlib_Avance.ipynb \
  --output-dir notebooks --ExecutePreprocessor.timeout=3600"
```

La primera ejecución convierte cada Excel por separado y guarda columnas seleccionadas en Parquet. Ejecuciones posteriores reutilizan esos archivos.

## Decisiones de reproducibilidad

- PySpark 3.5.1 y semilla global 42.
- `unionByName` para evitar errores por las 302 columnas de 2025T4.
- Esquema explícito durante la conversión Excel → Spark.
- Métricas sobre conjuntos completos; pandas solo recibe agregados o una muestra gráfica máxima de 5,000 filas.
- Exactamente seis predictores supervisados, sin identificadores, `FACTOR`, ingresos derivados ni clúster.
- Entrenamiento: 2025T1–T3; validación: 2025T4; test intacto: 2026T1.
- No se recorta ni transforma el salario objetivo en la comparación obligatoria.

## Fuente

Instituto Nacional de Estadística de Guatemala. *Encuesta Nacional de Empleo e Ingresos Continua (ENEIC)*, bases de Personas 2025–2026.
