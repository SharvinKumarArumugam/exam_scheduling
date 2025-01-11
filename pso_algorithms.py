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
        print(f"Error: File not found at path: {file_path}")
        return []  

# Now you can call load_data to load your CSV data:
students = load_data('/content/students.csv')
instructors = load_data('/content/instructors.csv')
courses = load_data('/content/courses.csv')
classrooms = load_data('/content/classrooms.csv')
timeslots = load_data('/content/timeslots.csv')


# Parameters for PSO
num_particles = 10
num_iterations = 100
w = 0.8  # Inertia weight
c1 = 1.5  # Cognitive component
c2 = 1.5  # Social component

# Initialize particles and velocities
particles = []
velocities = []

for _ in range(num_particles):
    particle = [(random.choice(courses), random.choice(instructors),
                 random.choice(classrooms), random.choice(timeslots))
                for _ in range(len(courses))]
    particles.append(particle)
    velocities.append([(0, 0, 0, 0) for _ in range(len(courses))])

# Fitness function
def fitness(particle):
    instructor_conflicts = len(particle) - len(set((c[1], c[3]) for c in particle))
    room_conflicts = len(particle) - len(set((c[2], c[3]) for c in particle))
    return instructor_conflicts + room_conflicts

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
            
            # Convert elements to numerical representations before subtraction
            p_best_num = [courses.index(p_best[i][j][0]), instructors.index(p_best[i][j][1]), 
                          classrooms.index(p_best[i][j][2]), timeslots.index(p_best[i][j][3])]
            particle_num = [courses.index(particle[j][0]), instructors.index(particle[j][1]),
                            classrooms.index(particle[j][2]), timeslots.index(particle[j][3])]
            
            cognitive = c1 * r1 * (np.array(p_best_num) - np.array(particle_num))
            
            # Convert g_best[j] elements to numerical representation as well
            g_best_num = [courses.index(g_best[j][0]), instructors.index(g_best[j][1]),
                          classrooms.index(g_best[j][2]), timeslots.index(g_best[j][3])]
            
            social = c2 * r2 * (np.array(g_best_num) - np.array(particle_num))  
            new_velocity = inertia + cognitive + social
            velocities[i][j] = new_velocity
        
        # Update position
        for j in range(len(particle)):
            # Get numerical representation for particle elements
            particle_num = [courses.index(particle[j][0]), instructors.index(particle[j][1]),
                            classrooms.index(particle[j][2]), timeslots.index(particle[j][3])]
            
            # Calculate new_position_num based on current position and velocity
            new_position_num = particle_num + velocities[i][j] 
            
            # Update position using numerical representation
            new_position_num = np.clip(new_position_num, 
                               [0, 0, 0, 0],  # Lower bounds for each dimension
                               [len(courses) - 1, len(instructors) - 1, 
                                len(classrooms) - 1, len(timeslots) - 1] # Upper bounds
                               ).astype(int)

            # Convert back to original representation
            particles[i][j] = (courses[new_position_num[0]],
                                instructors[new_position_num[1]],
                                classrooms[new_position_num[2]],
                                timeslots[new_position_num[3]])


        # Remove the problematic constraint application 
        #particles[i] = [(courses[int(p[0]) % len(courses)], # This line is causing the error
        #                 instructors[int(p[1]) % len(instructors)],
        #                 classrooms[int(p[2]) % len(classrooms)],
        #                 timeslots[int(p[3]) % len(timeslots)]) for p in particles[i]]
        
        # Evaluate fitness
        current_fitness = fitness(particles[i])
        if current_fitness < p_best_fitness[i]:
            p_best[i] = particles[i]
            p_best_fitness[i] = current_fitness
        
        if current_fitness < g_best_fitness:
            g_best = particles[i]
            g_best_fitness = current_fitness

    print(f"Iteration {iteration + 1}: Best Fitness = {g_best_fitness}")

print("\nOptimal Timetable:")
for course in g_best:
    print(f"Course {course[0]} - Instructor {course[1]} - Room {course[2]} - Timeslot {course[3]}")
