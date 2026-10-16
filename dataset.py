import numpy as np
import pandas as pd

def generate_student_data(n_samples=600, random_state=42):
    np.random.seed(random_state)
    
    study_hours_per_week = np.random.uniform(5, 35, n_samples)
    attendance_rate = np.random.uniform(50, 100, n_samples)
    past_sem_sgpa = np.random.uniform(4.0, 10.0, n_samples)
    internal_exam_marks = np.random.uniform(10, 30, n_samples)
    assignments_completed = np.random.randint(4, 11, n_samples)
    parental_education = np.random.choice([0, 1, 2], size=n_samples, p=[0.25, 0.50, 0.25])
    
    noise = np.random.normal(0, 3.5, n_samples)
    final_marks = (
        (study_hours_per_week * 0.8) +
        (attendance_rate * 0.25) +
        (past_sem_sgpa * 3.5) +
        (internal_exam_marks * 1.1) +
        (assignments_completed * 0.9) +
        (parental_education * 1.2) +
        noise
    )
    final_marks = np.clip(np.round(final_marks, 2), 0, 100)
    
    df = pd.DataFrame({
        'Study_Hours': np.round(study_hours_per_week, 1),
        'Attendance_Pct': np.round(attendance_rate, 1),
        'Past_SGPA': np.round(past_sem_sgpa, 2),
        'Internal_Marks': np.round(internal_exam_marks, 1),
        'Assignments_Completed': assignments_completed,
        'Parent_Edu_Level': parental_education,
        'Final_Marks': final_marks
    })
    return df

if __name__ == '__main__':
    df = generate_student_data()
    df.to_csv('student_data.csv', index=False)
    print("Dataset generated successfully as 'student_data.csv'!")