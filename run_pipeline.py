import subprocess

# Lista de scripts por paso, ubicados en la carpeta pipelines/
steps = [
    "pipelines/step_0_preprocess.py",
    "pipelines/step_1_merge_excels.py",
    "pipelines/step_2_generate_categories.py",
    "pipelines/step_3_clean_columns.py",
    "pipelines/step_4_transform_features.py",
    "pipelines/step_5_remove_relations.py",
    "pipelines/step_6_fpi_selection.py",
    "pipelines/step_7_train_models.py",
    "pipelines/step_8_prepare_output.py"
]

# Ejecuta cada paso secuencialmente
for step in steps:
    print(f"\n🔄 Ejecutando: {step}")
    result = subprocess.run(["python", step])
    if result.returncode != 0:
        print(f"❌ Error al ejecutar: {step}. Deteniendo el pipeline.")
        break
    print(f"✅ Finalizado: {step}")
