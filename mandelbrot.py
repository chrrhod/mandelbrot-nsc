"""
Mandelbrot Set Generator
Author : Christian Rhod
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time , statistics


def benchmark(func, *args, n_runs=3):
    """ Time func, return median of n_runs. """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
    
    median_t = statistics.median(times)
    print(f"Method: {func.__self__.__class__.__name__}, Median: {median_t:.4f}s (min={min(times):.4f}, max={max(times):.4f})")
    
    return median_t, result
    

class NaiveMandelbrot:
    def __init__(self, xmin, xmax, ymin, ymax, width, height, max_iter):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.max_iter = max_iter

    def mandelbrot_point(self, c):
        z = 0
        for n in range(self.max_iter):
            z = z*z + c
            if abs(z) > 2:
                return n 
        return self.max_iter

    def compute_mandelbrot(self):
        x = np.linspace(self.xmin, self.xmax, self.width)
        y = np.linspace(self.ymin, self.ymax, self.height)
        mandelbrot_set = np.zeros((self.height, self.width), dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                c = complex(x[j], y[i])
                mandelbrot_set[i, j] = self.mandelbrot_point(c)
        return mandelbrot_set

    def view_mandelbrot(self, mandelbrot_set):
        """
        Made with AI
        """
        plt.imshow(mandelbrot_set, extent=(self.xmin, xmax, ymin, ymax), cmap='inferno')
        plt.colorbar()
        plt.title('Mandelbrot Set')
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.show()

        #save the image
        plt.imsave('mandelbrot.png', mandelbrot_set, cmap='inferno')

    def get_image(self):
        mandelbrot_set = self.compute_mandelbrot()
        self.view_mandelbrot(mandelbrot_set)

class NumpyMandelbrot:
    def __init__(self, xmin, xmax, ymin, ymax, width, height, max_iter):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.max_iter = max_iter



    def compute_mandelbrot(self):
        x = np.linspace(self.xmin, self.xmax, self.width)
        y = np.linspace(self.ymin, self.ymax, self.height)

        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y

        mandelbrot_set = np.full(C.shape, self.max_iter, dtype=int)  # default = max_iter
        Z = np.zeros(C.shape, dtype=complex)
        active = np.ones(C.shape, dtype=bool)

        for n in range(self.max_iter):
            Z[active] = Z[active] * Z[active] + C[active]  
            diverged = active & (np.abs(Z) > 2)             
            mandelbrot_set[diverged] = n                    
            active[diverged] = False                      

        return mandelbrot_set
        

    def view_mandelbrot(self, mandelbrot_set):
        """
        Made with AI
        """
        plt.imshow(mandelbrot_set, extent=(self.xmin, xmax, ymin, ymax), cmap='inferno')
        plt.colorbar()
        plt.title('Mandelbrot Set')
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.show()

        #save the image
        plt.imsave('mandelbrot.png', mandelbrot_set, cmap='inferno')


    def get_image(self):
        mandelbrot_set = self.compute_mandelbrot()
        self.view_mandelbrot(mandelbrot_set)




if __name__ == "__main__":
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100

    run_naive = True
    run_numpy = True

    times = {}
    results = {}

    if run_naive:
        Naive = NaiveMandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
        naive_t, naive_M = benchmark(Naive.compute_mandelbrot)
        times["Naive"] = naive_t
        results["Naive"] = naive_M

    if run_numpy:
        Numpy = NumpyMandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
        numpy_t, numpy_M = benchmark(Numpy.compute_mandelbrot)
        times["Numpy"] = numpy_t
        results["Numpy"] = numpy_M

    print("\nPerformance gains:")
    for method, t in times.items():
        if method == "Naive":
            continue
        gain = times["Naive"] / t
        print(f"{method}: {gain:.2f}x faster than Naive")
    



    #check if the results are the same
    if run_naive and run_numpy:
        if np . allclose (results["Naive"], results["Numpy"]):
            print (" Results match !")
        else:
            print (" Results differ !")
    
    # Check where they differ :
    diff = np .abs ( results["Naive"] - results["Numpy"] )
    print (f" Max difference : { diff . max ()}")
    print (f" Different pixels : {( diff > 0). sum ()}")



    #Naive.get_image()
    Numpy.get_image()