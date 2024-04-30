import numpy as np
import matplotlib.pyplot as plt
import argparse


def calculate_agreement(population, row, col, external=0.0):
    
    n_row, n_col = population.shape         #give number of row and coloum by population
    neighbors = [((row-1)% n_row,col),((row+1)% n_row,col),(row,(col-1)% n_col),(row,(col+1)% n_col)]      
    #represent a circular or wrap-around boundary condition for accessing elements of a 2D grid

    neighbor = [population[i] for i in neighbors]
    opinion = population[row][col]      #opinion of myself
    agreement = opinion*np.sum(neighbor) + opinion * external       #equation of agreement
    
    return agreement

def ising_step(population, alpha=None, external=0.0):
    '''
	This function will perform a single update of the Ising model
	Inputs: population (numpy array)
			external (float) - optional - the magnitude of any external "pull" on opinion
	'''

    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)          #choose random row and coloumn
    col = np.random.randint(0, n_cols)          
    agreement = calculate_agreement(population, row, col, external)
    if agreement <= 0:
        population[row,col] *= -1
    #agreement less and equal than 0, change opinion

    elif agreement > 0 and alpha is not None and alpha != 0:
        prob_change = np.exp(- agreement / alpha)
        if np.random.rand() < prob_change:
            population[row,col] *= -1
            
    

def plot_ising(im, population):

    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population],dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)

def test_ising():
    '''
	This function will test the calculate_agreement function in the Ising model
	'''
    print('Testing isin model calculations')
    population = -np.ones((3,3))
    assert(calculate_agreement(population,1,1) == 4), "Test 1"

    population[1,1] = 1.
    assert(calculate_agreement(population,1,1) == -4), "Test 2"
    
    population[0,1] = 1.
    assert(calculate_agreement(population,1,1) == -2), "Test 3"

    population[1,0] = 1.
    assert(calculate_agreement(population,1,1) == 0), "Test 4"

    population[2,1] = 1.
    assert(calculate_agreement(population,1,1) == 2), "Test 5"

    population[1,2] = 1.
    assert(calculate_agreement(population,1,1) == 4), "Test 6"

    "Testing external pull"
    population = -np.ones((3,3))
    assert(calculate_agreement(population,1,1,1) == 3), "Test 7"
    assert(calculate_agreement(population,1,1,-1) == 5), "Test 8"
    assert(calculate_agreement(population,1,1,-10) == 14), "Test 9"
    assert(calculate_agreement(population,1,1,10) == -6), "Test 10"

    print("Tests passed")

def ising_main(population,alpha=None,external=0.0):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im= ax.imshow(population, interpolation='none', cmap='RdPu_r')

    for frame in range(100):
        for step in range(1000):
            ising_step(population,alpha, external)
        print('Step:', frame, end='\r')
        plot_ising(im,population)
    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ising_model', action = 'store_true')
    parser.add_argument('-external', default=0, type=float)
    parser.add_argument('-alpha', default=1, type=float)
    parser.add_argument('-test_ising', action = 'store_true')

    args = parser.parse_args()
    if args.ising_model:
        alpha=args.alpha
        external=args.external
        population = np.random.choice([-1, 1], size=(100, 100))
        ising_main(population, alpha, external)

    if args.test_ising:
        print(test_ising())


if __name__ == "__main__":
    main()

    

        
    



