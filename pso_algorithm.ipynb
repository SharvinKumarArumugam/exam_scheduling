import streamlit as st
import random
import numpy as np
import pandas as pd

# Function to load data from a CSV file
def load_data(file_path):
    """Loads data from a CSV file and returns a list of values from the first column."""
    try:
        df = pd.read_csv(file_path)
        return df.iloc[:, 0].tolist()  
    except FileNotFoundError:
        st.error(f"Error: File not found at path: {file_path}")
        return []  

# Parameters for PSO
num_particles = 10
num_iterations = 100
w = 0.8  # Inertia weight
c1 = 1.5  # Cognitive component
c2 = 1.5  # Social component

# Fitness function
def fitness(particle):
    instructor_conflicts = len(particle) - len(set((c[1], c[3]) for c in particle))
    room_conflicts = len(particle) - len(set((c[2], c[3]) for c in particle))
    return instructor_conflicts + room_conflicts

# Streamlit UI
st.title('Timetable Optimization using PSO')
st.sidebar.header("Upload Your CSV Files")

students_file = st.sidebar.file_uploader("Upload Students CSV", type=["csv"])
instructors_file = st.sidebar.file_uploader("Upload Instructors CSV", type=["csv"])
courses_file = st.sidebar.file_uploader("Upload Courses CSV", type=["csv"])
classrooms_file = st.sidebar.file_uploader("Upload Classrooms CSV", type=["csv"])
timeslots_file = st.sidebar.file_uploader("Upload Timeslots CSV", type=["csv"])

if students_file and instructors_file and courses_file and classrooms_file and timeslots_file:
    # Load the data after the user uploads the files
    students = load_data(students_file)
    instructors = load_data(instructors_file)
    courses = load_data(courses_file)
    classrooms = load_data(classrooms_file)
    timeslots = load_data(timeslots_file)

    # Initialize particles and velocities
    particles = []
    velocities = []

    for _ in range(num_particles):
        particle = [(random.choice(courses), random.choice(instructors),
                     random.choice(classrooms), random.choice(timeslots))
                    for _ in range(len(courses))]
        particles.append(particle)
        velocities.append([(0, 0, 0, 0) for _ in range(len(courses))])

    # Personal and global bests
    p_best = particles.copy()
    p_best_fitness = [fitness(p) for p in particles]
    g_best = particles[np.argmin(p_best_fitness)]
    g_best_fitness = min(p_best_fitness)

    # PSO main loop
    for iteration in range(num_iterations):
        for i, particle in enumerate(particles):
            # Update velocity
            for j in range(len(particle)):
                r1, r2 = random.random(), random.random()
                inertia = np.array(velocities[i][j]) * w
                
                p_best_num = [courses.index(p_best[i][j][0]), instructors.index(p_best[i][j][1]), 
                              classrooms.index(p_best[i][j][2]), timeslots.index(p_best[i][j][3])]
                particle_num = [courses.index(particle[j][0]), instructors.index(particle[j][1]),
                                classrooms.index(particle[j][2]), timeslots.index(particle[j][3])]
                
                cognitive = c1 * r1 * (np.array(p_best_num) - np.array(particle_num))
                
                g_best_num = [courses.index(g_best[j][0]), instructors.index(g_best[j][1]),
                              classrooms.index(g_best[j][2]), timeslots.index(g_best[j][3])]
                
                social = c2 * r2 * (np.array(g_best_num) - np.array(particle_num))  
                new_velocity = inertia + cognitive + social
                velocities[i][j] = new_velocity
            
            # Update position
            for j in range(len(particle)):
                particle_num = [courses.index(particle[j][0]), instructors.index(particle[j][1]),
                                classrooms.index(particle[j][2]), timeslots.index(particle[j][3])]
                
                new_position_num = particle_num + velocities[i][j]
                
                new_position_num = np.clip(new_position_num, 
                                   [0, 0, 0, 0],  
                                   [len(courses) - 1, len(instructors) - 1, 
                                    len(classrooms) - 1, len(timeslots) - 1]
                                   ).astype(int)

                particles[i][j] = (courses[new_position_num[0]],
                                    instructors[new_position_num[1]],
                                    classrooms[new_position_num[2]],
                                    timeslots[new_position_num[3]])

            # Evaluate fitness
            current_fitness = fitness(particles[i])
            if current_fitness < p_best_fitness[i]:
                p_best[i] = particles[i]
                p_best_fitness[i] = current_fitness
            
            if current_fitness < g_best_fitness:
                g_best = particles[i]
                g_best_fitness = current_fitness

        st.write(f"Iteration {iteration + 1}: Best Fitness = {g_best_fitness}")

    st.write("### Optimal Timetable:")
    timetable_df = pd.DataFrame(g_best, columns=["Course", "Instructor", "Room", "Timeslot"])
    st.dataframe(timetable_df)

else:
    st.warning("Please upload all the necessary CSV files.")
