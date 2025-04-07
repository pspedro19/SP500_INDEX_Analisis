import subprocess

steps = [
    "preprocess_step_0.py",
    "merge_excels_step_1.py",
    "generate_categories_step_2.py",
    "clean_columns_step_3.py",
    "run_pipeline_step_4.py"
    "Eliminar_relaciones_step_5.py"
    "FPI_Feature_Section_step_6.py"
    "Training.py"
]

for step in steps:
    print(f"🔄 Ejecutando: {step}")
    result = subprocess.run(["python", step])
    if result.returncode != 0:
        print(f"❌ Error al ejecutar: {step}")
        break
    print(f"✅ Finalizado: {step}\n")
