import matplotlib.pyplot as plt

def plot_age_histogram(deleted_boids):
    age = []
    for boid in deleted_boids:
        age.append(boid.age)
    plt.hist(age)