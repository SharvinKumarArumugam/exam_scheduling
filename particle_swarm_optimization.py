import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Load datasets
schedule_df = pd.read_csv('schedule.csv')
courses_df = pd.read_csv('courses.csv')
timeslots_df = pd.read_csv('timeslots.csv')
classrooms_df = pd.read_csv('classrooms.csv')

# Merge datasets
exam_timetable = schedule_df.merge(courses_df, on='course_id').merge(timeslots_df, on='timeslot_id')

# Select and rename relevant columns
exam_timetable = exam_timetable[['course_name', 'day', 'start_time', 'end_time', 'classroom_id']]
exam_timetable.columns = ['Subject', 'Date', 'Start Time', 'End Time', 'Venue']

# Streamlit app title
st.title("Optimized Exam Timetable Generator")

# Display the original exam timetable
st.subheader("Original Exam Timetable")
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
st.dataframe(optimal_timetable_df)

# Visualize the optimized exam timetable
fig, ax = plt.subplots(figsize=(10, 6))

# Convert 'Start Time' and 'End Time' to datetime objects
exam_timetable['Start Time'] = pd.to_datetime(exam_timetable['Start Time'], format='%H:%M')
exam_timetable['End Time'] = pd.to_datetime(exam_timetable['End Time'], format='%H:%M')

# Plot each exam slot as a horizontal bar
for i, row in optimal_timetable_df.iterrows():
    start_time = pd.to_datetime(exam_timetable.loc[exam_timetable['Subject'] == row['Subject'], 'Start Time'].values[0], format='%H:%M')
    end_time = pd.to_datetime(exam_timetable.loc[exam_timetable['Subject'] == row['Subject'], 'End Time'].values[0], format='%H:%M')
    ax.barh(row['Subject'], (end_time - start_time).total_seconds() / 3600,
            left=start_time.hour + start_time.minute / 60, height=0.5)

# Set labels and title
ax.set_xlabel('Time (Hours)')
ax.set_ylabel('Subjects')
ax.set_title('Optimized Exam Timetable')

# Show the plot in Streamlit
st.pyplot(fig)

import pandas as pd
import streamlit as st

# Assuming your existing PSO code is already present here

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
