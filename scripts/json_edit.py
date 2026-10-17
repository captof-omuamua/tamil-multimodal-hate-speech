import json

# Load the JSON file
with open("/home/raptor/Desktop/sem_7/project_phase_1/teacher_student_model/HATE_SPEECH_FINETUNE.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Save with double quotes in a new file
with open("/home/raptor/Desktop/sem_7/project_phase_1/teacher_student_model/HATE_SPEECH_FINETUNE_mod.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=2)
