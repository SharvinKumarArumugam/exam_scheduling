import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Function to load uploaded CSV file
def load_csv(file):
    return pd.read_csv(file)

# Streamlit UI
st.title("Optimized Exam Timetable Generation using PSO")

# File upload for schedule, courses, timeslots, and classrooms
st.subheader("Upload Schedule CSV")
schedule_file = st.file_uploader("Upload Schedule CSV", type=["csv"])

st.subheader("Upload Courses CSV")
courses_file = st.file_uploader("Upload Courses CSV", type=["csv"])

st.subheader("Upload Timeslots CSV")
timeslots_file = st.file_uploader("Upload Timeslots CSV", type=["csv"])

st.subheader("Upload Classrooms CSV")
classrooms_file = st.file_uploader("Upload Classrooms CSV", type=["csv"])

# Process the data if all files are uploaded
if schedule_file and courses_file and timeslots_file and classrooms_file:
    schedule_df = load_csv(schedule_file)
    courses_df = load_csv(courses_file)
    timeslots_df = load_csv(timeslots_file)
    classrooms_df = load_csv(classrooms_file)
    
    # Merge datasets
    exam_timetable = schedule_df.merge(courses_df, on='course_id').merge(timeslots_df, on='timeslot_id')

    # Select and rename relevant columns
    exam_timetable = exam_timetable[['course_name', 'day', 'start_time', 'end_time', 'classroom_id']]
    exam_timetable.columns = ['Subject', 'Date', 'Start Time', 'End Time', 'Venue']

    # Display the exam timetable
    st.write("Generated Exam Timetable:")
    st.dataframe(exam_timetable)

    # Parameters for PSO
    num_particles = 10
    num_iterations = 100
    w = 0.8  # Inertia weight
    c1 = 1.5  # Cognitive component
    c2 = 1.5  # Social component

    # Initialize particles and velocities
    particles = []
    velocities = []

    # Initialize particles with random assignments of courses to timeslots and classrooms
    for _ in range(num_particles):
        particle = []
        for idx, row in exam_timetable.iterrows():
            timeslot = random.choice(timeslots_df['timeslot_id'])
            classroom = random.choice(classrooms_df['classroom_id'])
            particle.append((row['Subject'], timeslot, classroom))
        particles.append(particle)
        velocities.append([(0, 0) for _ in range(len(exam_timetable))])

    # Fitness function
    def fitness(particle):
        conflicts = 0
        # Track instructor conflicts and classroom conflicts
        timeslot_map = {}
        classroom_map = {}
        
        for course, timeslot, classroom in particle:
            # Check for timeslot conflicts
            if timeslot in timeslot_map:
                conflicts += 1
            timeslot_map[timeslot] = timeslot_map.get(timeslot, 0) + 1
            
            # Check for classroom conflicts
            if classroom in classroom_map:
                conflicts += 1
            classroom_map[classroom] = classroom_map.get(classroom, 0) + 1

        return conflicts

    # Personal and global bests
    p_best = particles.copy()
    p_best_fitness = [fitness(p) for p in particles]
    g_best = particles[np.argmin(p_best_fitness)]
    g_best_fitness = min(p_best_fitness)

    # PSO main loop
    fitness_trends = []
    for iteration in range(num_iterations):
        for i, particle in enumerate(particles):
            # Update velocity
            for j in range(len(particle)):
                r1, r2 = random.random(), random.random()
                inertia = np.array(velocities[i][j]) * w

                # Calculate cognitive and social components
                p_best_num = [timeslots_df['timeslot_id'].tolist().index(p_best[i][j][1]),
                              classrooms_df['classroom_id'].tolist().index(p_best[i][j][2])]
                particle_num = [timeslots_df['timeslot_id'].tolist().index(particle[j][1]),
                                classrooms_df['classroom_id'].tolist().index(particle[j][2])]

                cognitive = c1 * r1 * (np.array(p_best_num) - np.array(particle_num))
                g_best_num = [timeslots_df['timeslot_id'].tolist().index(g_best[j][1]),
                              classrooms_df['classroom_id'].tolist().index(g_best[j][2])]
                social = c2 * r2 * (np.array(g_best_num) - np.array(particle_num))

                new_velocity = inertia + cognitive + social
                velocities[i][j] = new_velocity

            # Update position
            for j in range(len(particle)):
                particle_num = [timeslots_df['timeslot_id'].tolist().index(particle[j][1]),
                                classrooms_df['classroom_id'].tolist().index(particle[j][2])]
                new_position_num = np.array(particle_num) + velocities[i][j]
                # Ensure new_position_num is an integer for iloc indexing
                new_position_num = np.clip(new_position_num.astype(int), [0, 0], [len(timeslots_df) - 1, len(classrooms_df) - 1])
                particles[i][j] = (particle[j][0], timeslots_df['timeslot_id'].iloc[new_position_num[0]],
                                    classrooms_df['classroom_id'].iloc[new_position_num[1]]) 

            # Evaluate fitness
            current_fitness = fitness(particles[i])
            if current_fitness < p_best_fitness[i]:
                p_best[i] = particles[i]
                p_best_fitness[i] = current_fitness

            if current_fitness < g_best_fitness:
                g_best = particles[i]
                g_best_fitness = current_fitness

        fitness_trends.append(g_best_fitness)
        st.write(f"Iteration {iteration + 1}: Best Fitness = {g_best_fitness}")

    # Output the optimal timetable
    st.subheader("Optimal Exam Timetable")
    optimal_timetable = []
    for course, timeslot, classroom in g_best:
        optimal_timetable.append({
            'Subject': course,
            'Timeslot': timeslot,
            'Classroom': classroom
        })

    optimal_timetable_df = pd.DataFrame(optimal_timetable)
    st.write("Optimal Timetable Data:")
    st.dataframe(optimal_timetable_df)

    # Merge with timeslots_df and classrooms_df to get details
    final_timetable = pd.merge(optimal_timetable_df, timeslots_df, left_on='Timeslot', right_on='timeslot_id')
    final_timetable = pd.merge(final_timetable, classrooms_df, left_on='Classroom', right_on='classroom_id')

    # Select and rename columns for the final table
    final_timetable = final_timetable[['Subject', 'day', 'start_time', 'end_time', 'building_name', 'room_number']]
    final_timetable.columns = ['Subject', 'Day', 'Start Time', 'End Time', 'Building', 'Room']

    # Display the final exam timetable in a table format
    st.subheader("Final Exam Timetable")
    st.dataframe(final_timetable)

    # Plot the fitness trend over generations
    st.subheader("Fitness Trend Over Generations")
    plt.figure(figsize=(12, 6))
    plt.plot(fitness_trends, label="Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Fitness Trends Over Generations")
    plt.legend()
    st.pyplot()

else:
    st.write("Please upload all the required CSV files.")
