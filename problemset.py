#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:02:13 2020
@author: martin

Part of the Data-driven optimization approach

This file will create optimization testfunctions
"""

# libraries
import numpy as np
from scipy.stats import ortho_group
import SALib.sample.latin as slb

import pygmo as pg


DIMENSIONALITY = 2

class Function:
    """
    Main class to initialize a test function
    """

    def __init__(self, seed, name, orig_bounds=[0.0, 1.0], char=[], dim=DIMENSIONALITY):
        self.name = name
        "Name of the function: f + seed."

        self.dim = dim
        "Dimensionality of the testproblem. Default is 2."

        self.pop = self.dim * 2 + 1  # arbitrary
        "Population size of the initial solutions."

        self.bounds = [0.0, 1.0]  # normalized bounds
        "Normalized box-constrained bounds of the testproblem."

        self.orig_bounds = orig_bounds

        self.seed = seed
        "Random number generator seed."
        np.random.seed(seed=seed)

        self.reroll_counter = 0
        "Takes track of how many times rerolls are requested"

        self.fevals = 0
        "Takes track of the number of function evaluations"

        # Rotation, offset and noise parameters

        self.o = np.random.uniform(0.0, 0.5, self.dim)  # offset
        "Input parameter off-set."

        self.m = self.setrotationmatrix(
            np.random.choice([0.0, np.random.uniform(1, 360)])
        )
        # self.m = np.identity(self.dim)
        "Input parameter rotation."

        self.sigma = 0.0  # initially without noise
        "Standard deviation of objective value noise."

        # self.sigma = np.random.choice([0.,np.random.uniform(0.1,1.0)])  #noise variance

        self.char = char
        "List of features on the testproblem."

        self.getx0()

    def reset_seed(self):
        np.random.seed(seed=self.seed)
        self.reroll_counter = 0
        self.fevals = 0
        self.setx0(forcereset=True)

    def get_seed(self):
        return self.seed

    def setrotationmatrix(self, degree):
        """
        Generate a rotation matrix in the appropriate dimensionality

        Parameters
        ----------
        degree : float
            Rotation in degrees

        Returns
        -------
        R : numpy array
            Rotation matrix
        """
        theta = np.radians(degree)
        c, s = np.cos(theta), np.sin(theta)
        if self.dim == 2:  # 2D rotation
            R = np.array(((c, -s), (s, c)))

        if self.dim == 3:  # 3D rotation
            R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

        if self.dim > 3:  # nD: no rotation, or orthogonal matrix
            # R = np.identity(self.dim)
            R = ortho_group.rvs(dim=self.dim)
        return R

    def rotation(self, x, rotation_point=0.0):
        """
        Transform the input with the given rotation matrix

        Parameters
        ----------
        x : np.array
            Input parameters.
        rotation_points : float, optional
            Point (x,y) to set as centre for rotating. Standard = origin

        Returns
        -------
            Transformed input parameters.
        """
        return (
            np.linalg.solve(self.m, x - np.ones(self.dim) * rotation_point)
            + np.ones(self.dim) * rotation_point
        )

    def offset(self, x):
        return x - self.o

    def noise(self, y):
        np.random.seed(self.seed + self.fevals)
        yn = np.random.normal(loc=0.0, scale=abs(self.sigma * y), size=None)
        np.random.seed(self.seed)
        return y + yn  # this doesnt comply to the seed

    def denormalize(self, x):
        return (self.orig_bounds[1] - self.orig_bounds[0]) * x + self.orig_bounds[0]

    def ask(self, x, skip_fevals=False):
        x = self.rotation(x)
        x = self.offset(x)
        x = self.denormalize(x)

        y = self.func(x)

        y = self.noise(y)

        if not skip_fevals:
            self.fevals += 1

        return y

    def makedictx0(self):
        "Make dictionary for SALib.sample"
        names = []
        b = []
        for i in range(self.dim):
            names.append("x%i" % i)

        for j in range(self.dim):
            b.append(self.bounds)

        problem = {"num_vars": self.dim, "names": names, "bounds": b}
        return problem

    def getx0(self):
        # "Random sampling"
        # self.x0 = np.random.uniform(0.,1.,size=(self.pop,self.dim))

        "Latin hypercube sampling using SALib.sample"
        problem = self.makedictx0()
        self.x0 = slb.sample(problem, self.pop, seed=self.seed)

        "Compute objective values of x0"
        self.y0 = np.array([self.ask(i) for i in self.x0])

        "Set variance of noise proportional to variance of surface"
        low_noise = np.random.uniform(low=0.005, high=0.01)
        medium_noise = np.random.uniform(low=0.05, high=0.07)
        high_noise = np.random.uniform(low=0.10, high=0.20)
        self.sigma = np.random.choice(
            [0.0, low_noise, medium_noise, high_noise]
        )  # 0.05,0.01 np.random.normal(loc=0, scale=0.005)

        np.random.seed(self.seed)
        self.fevals = 0
        self.y0 = np.array([self.ask(i) for i in self.x0])

        if self.sigma == 0.0:
            self.char.append("smooth")

        else:
            self.char.append("noisy")

    def setx0(self, forcereset=False):

        if forcereset:
            self.reroll_counter = 0
        else:
            self.reroll_counter += 1

        problem = self.makedictx0()
        self.x0 = slb.sample(problem, self.pop, seed=self.seed + self.reroll_counter)
        np.random.seed(self.seed)
        self.fevals = 0
        self.y0 = np.array([self.ask(i) for i in self.x0])

    def plot3d(self, px=100, ax=None):
    
        """
        Generate a 3D plot of a slice of the 4D function
        px = number of evaluations in one dimension for plotting the response surface
        """
        # import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcol
    
        plt.style.use("PaperDoubleFig.mplstyle")
        fevals_before = self.fevals
    
        X1 = np.linspace(self.bounds[0], self.bounds[1], num=px)
        X2 = np.linspace(self.bounds[0], self.bounds[1], num=px)
        X1, X2 = np.meshgrid(X1, X2)
    
        X = np.zeros(self.dim)
        Y = np.zeros([len(X1), len(X1)])
    
        for i in range(len(X1)):
            for j in range(len(X1)):
                X[0] = X1[i, j]
                X[1] = X2[i, j]
                Y[i, j] = self.ask(X)
    
        # normalize Y
        Ymin = Y.min()
        Ymax = Y.max()
        for i in range(len(X1)):
            for j in range(len(X1)):
                Y[i, j] = (Y[i, j] - Ymin) / (Ymax - Ymin)
    
        fig = plt.figure(figsize=(10, 10), constrained_layout=True)
        if ax == None:
            ax = plt.axes(projection="3d", elev=50, azim=-50)  # elev=50, azim=-50
    
        norm = mcol.LogNorm()
        # norm = mcol.Normalize()
    
        ax.plot_surface(
            X1,
            X2,
            Y,
            rstride=1,
            cstride=1,
            edgecolor="none",
            alpha=0.8,
            cmap="viridis",
            norm=norm,
            zorder=1,
        )  # 0.8
    
        # ax.set_xlabel('$X_{1}$',fontsize='small') #'x-large'
        # ax.set_ylabel('$X_{2}$',fontsize='small')
        # ax.set_zlabel('$f(X)$',fontsize='small')
    
        ax.set_xlim((self.bounds[0], self.bounds[1]))
        ax.set_ylim((0.0, 1.0))
    
        ax.set_xticks(np.linspace(self.bounds[0], self.bounds[1], 11))
        ax.set_yticks(np.linspace(self.bounds[0], self.bounds[1], 11))
        ax.set_zticks(np.linspace(0.0, 1.0, 11))
    
        ax.set_xticklabels(())
        ax.set_yticklabels(())
        ax.set_zticklabels(())
    
        ax.tick_params(axis="y", which="both", labelsize="small")  #'large'
        ax.tick_params(axis="x", which="both", labelsize="small")
        ax.tick_params(axis="z", which="both", labelsize=None)
    
        self.fevals = fevals_before
        return fig, ax


    def plot2d(self, data=None, algo="CMAES", px=300, reroll=0, points=None, ax=None):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcol
    
        X1 = np.linspace(0, 1, num=px)
        X2 = np.linspace(0, 1, num=px)
        X1, X2 = np.meshgrid(X1, X2)
    
        Y = np.zeros([len(X1), len(X1)])
    
        for i in range(len(X1)):
            for j in range(len(X1)):
                xy = np.array([X1[i, j], X2[i, j]])
                Y[i, j] = self.ask(xy)
    
        # normalize Y
        Ymin = Y.min()
        Ymax = Y.max()
        for i in range(len(X1)):
            for j in range(len(X1)):
                xy = np.array([X1[i, j], X2[i, j]])
                Y[i, j] = (Y[i, j] - Ymin) / (Ymax - Ymin)
    
        norm = mcol.LogNorm()
    
        fig = plt.figure(figsize=(7, 7), constrained_layout=True)
        if ax == None:
            ax = plt.axes()
    
        ax.set_xlim(0, px)
        ax.set_ylim(0, px)
        ax.set_frame_on(False)
        ax.tick_params(
            which="both", bottom=False, left=False, labelbottom=False, labelleft=False
        )
        ax.imshow(Y, cmap="viridis", norm=norm)
    
        color_dict = {
            "Adam": "Greens",
            "PSO": "Purples",
            "CMAES": "Oranges",
            "BayesianOpt": "Reds",
            "RandomSearch": "Blues",
            "DataOpt": "Greys",
            "LBFGS": "spring",
            "SGA": "Blues",
        }
    
        color_opt = {
            "Adam": "green",
            "PSO": "purple",
            "CMAES": "orange",
            "BayesianOpt": "red",
            "RandomSearch": "blue",
            "DataOpt": "grey",
            "LBFGS": "pink",
            "SGA": "blue",
        }
    
        def algo_length(algolist):
            a = algolist[-1]
            indexlist = np.where(np.array(algolist) == a)[0]
            streak = indexlist[-1]
            for i in reversed(indexlist[:-1]):
                if i == streak - 1:
                    streak = i
                else:
                    break
            return streak
    
        if data != None:
            keys = list(data.keys())
            xi = np.array(data[algo][reroll]["x"][:points])
            yi = np.array(data[algo][reroll]["y"][:points])
            if points != 0:  # len(data[algo][reroll]['y'])
                ax.scatter(
                    xi[:, 0] * px,
                    xi[:, 1] * px,
                    c=np.linspace(0, 1, 200)[:points],
                    cmap=color_dict[algo],
                    edgecolors="black",
                    label="_nolegend_",
                    s=30,
                    norm=mcol.Normalize(vmin=0, vmax=1),
                )  #
    
                streak = algo_length(data[algo][0]["algo"])
                ax.scatter(
                    xi[:streak, 0] * px,
                    xi[:streak, 1] * px,
                    color="white",
                    edgecolors="red",
                    label="Starting parameters",
                )
    
            xbest = xi[np.argmin(yi)]
    
            # ax.scatter(xbest[0]*px,xbest[1]*px,c=color_opt[algo],edgecolors='black',marker='X',s=0.55*px,label=algo+' optimum')
        Ym = np.unravel_index(Y.argmin(), Y.shape)
    
        ax.scatter(
            Ym[1],
            Ym[0],
            color="white",
            edgecolors="black",
            marker="X",
            s=1.5 * px,
            label="Global optimum",
            alpha=0.8,
        )  # 0.55*px
        ax.set_xlabel("$X_{1}$", fontsize=16)  # 20
        ax.set_ylabel("$X_{2}$", fontsize=16)
        ax.yaxis.set_label_position("right")
        ax.legend(fontsize="small", loc="lower right")  #'x-large'
        return fig, ax

class Ronkkonen(Function):
    """
    Hyperparameters and attributes of the tunable testfunctions
    """

    def __init__(self, seed, name, char=[], orig_bounds=[0.0,1.0]):
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)
        # orig_bounds = [0.0, 1.0]

        # enable rotation
        self.m = ortho_group.rvs(dim=self.dim)

        "Hyperparameters of Ronkonen functions"
        ## f_multimodal
        while True:  # make sure that _g is not all ones. This will break the function.
            self._g = np.random.randint(1.0, 5.0, self.dim)
            if not (self._g == np.ones(self.dim, dtype=int)).all():
                break

        self._l = np.random.randint(0.0, 7.0, self.dim)  # number of local minima
        self._alpha = np.random.uniform(0.0, 1.0)  # shape

        ## f_quadratic
        self._q = np.random.randint(3.0, 10.0)  # amount of minima
        self._p = np.random.uniform(0.0, 1.0, (self._q, self.dim))  # position of minima

        self._v = np.random.uniform(-5.0, -3, self._q)  # values of minima
        self._b = np.random.uniform(3.0, 20.0, self._q)  # shape of minima
        self._order = np.random.choice([2.0, 4], self._q)  # shape of bowl

        ## f_plate
        self._slope = np.random.uniform(-1.0, 1.0, (self._q, self.dim))
        self._intersect = np.random.uniform(-0.5, 0.5, (self._q, self.dim))

        ## f_steep
        self._c = np.random.uniform(
            -100.0, -3, self._q
        )  # value of minima of steep function
        self._r = np.random.uniform(0.05, 0.3, self._q)  # 0.3,0.5
        self._w = np.random.uniform(0.3, 3.0, self._q)  # 0.0,0.3

    def f_multimodal(self, x):
        "Cosine function"
        # minimum = -2 -2*alpha

        a = -np.cos((self._g - 1) * 2 * np.pi * x) - self._alpha
        b = np.cos((self._g - 1) * 2 * np.pi * self._l * x)
        c = sum(a * b) / 2 * self.dim + (2 ** (self.dim - 1)) * (1 + self._alpha)

        return c

    def f_bowl(self, x):
        "Quadratic centre function"
        a = 20 * (x) ** 2
        c = sum(a)
        return c

    def f_quad(self, x):
        "Combination of different x^n-functions"
        c = np.zeros(self._q)
        for i in range(self._q):
            xx = x - self._p[i]
            for j in range(int(self._order[i] - 1)):
                xx = xx * (x - self._p[i])

            c[i] = sum(xx * self._b[i] + self._v[i])

        return min(c)

    def f_plate(self, x):
        "Combination of linear plate functions"

        c = np.zeros(self._q)
        for i in range(self._q):
            c[i] = sum(x * self._slope[i] + self._intersect[i])

        return np.sort(c)[-1]

    def f_steep(self, x):
        "Steep drops"
        c = np.zeros(self._q)
        for i in range(self._q):
            dist = np.linalg.norm(x - self._p[i])
            if dist <= self._r[i]:
                c[i] = self._c[i] * (1.0 - (dist / self._r[i]) ** self._w[i])

            else:
                c[i] = 1.0

        return min(c)


# ......................................

# Create a list with all functions to choose from"
funclist = []

# ......................................
"""
Generated tunable test functions
from Ronkkonen, 2010
"""
# ......................................


class TuneMultimodal(Ronkkonen):
    def __init__(self, seed, name):
        char = ["multimodal"]
        Ronkkonen.__init__(self, seed, name, char=char)

    def func(self, x):
        c = self.f_multimodal(x)
        return c


funclist.append(TuneMultimodal)

# ......................................


class TuneBowl(Ronkkonen):
    def __init__(self, seed, name):
        char = ["unimodal"]
        Ronkkonen.__init__(self, seed, name, char=char)

    def func(self, x):
        c = self.f_bowl(x)
        return c


funclist.append(TuneBowl)
# ......................................


class TuneMultimodalBowl(Ronkkonen):
    def __init__(self, seed, name):
        char = ["multimodal"]
        Ronkkonen.__init__(self, seed, name, char=char)

    def func(self, x):
        c = 5 * self.f_multimodal(x) + self.f_bowl(x)
        return c


funclist.append(TuneMultimodalBowl)
# ......................................


class TuneQuad(Ronkkonen):
    def __init__(self, seed, name):
        char = ["multimodal"]
        super().__init__(seed, name, char=char)

    def func(self, x):
        c = self.f_quad(x)
        return c


funclist.append(TuneQuad)
# ......................................


class TunePlate(Ronkkonen):
    def __init__(self, seed, name):
        char = ["unimodal"]
        Ronkkonen.__init__(self, seed, name, char=char)

    def func(self, x):
        c = self.f_plate(x)
        return c


funclist.append(TunePlate)

# ......................................


class TuneSteep(Ronkkonen):
    def __init__(self, seed, name):
        char = ["steep"]
        Ronkkonen.__init__(self, seed, name, char=char)

        # disable rotation and offset
        # this is to make sure the minima are not outside the bounds and the response surface is entirely flat
        self.o = np.zeros(self.dim, dtype=float)
        self.m = np.identity(self.dim)
        # self.getx0()

    def func(self, x):
        c = self.f_steep(x)
        return c


funclist.append(TuneSteep)

# ......................................


class TuneMix(Ronkkonen):
    def __init__(self, seed, name):
        self.ratio = np.random.uniform(0.0, 1.0, 4)
        Ronkkonen.__init__(self, seed, name, char=[])
        # self.getx0()

    def func(self, x):
        c = 0
        c += self.f_multimodal(x) * self.ratio[0]
        c += self.f_bowl(x) * self.ratio[1]
        c += self.f_plate(x) * self.ratio[2]
        c += self.f_steep(x) * self.ratio[3]
        return c


funclist.append(TuneMix)

# ......................................
"""
Benchmark optimization functions
from https://www.sfu.ca/~ssurjano/optimization.html
"""
# ......................................


class Levy(Function):
    def __init__(self, seed, name):

        orig_bounds = [-10.0, 10.0]
        char = ["multimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):
        z = 1 + (x - 1) / 4
        c = (
            np.sin(np.pi * z[0]) ** 2
            + sum((z[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * z[:-1] + 1) ** 2))
            + (z[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * z[-1]) ** 2)
        )
        return c


funclist.append(Levy)

# ......................................


class Ackley(Function):
    def __init__(self, seed, name):
        orig_bounds = [-40.0, 40.0]
        char = ["multimodal", "steep"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x, a=20, b=0.2, c=2 * np.pi):
        n = len(x)

        s1 = sum(x**2)
        s2 = sum(np.cos(c * x))
        cc = -a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.exp(1)
        return cc


funclist.append(Ackley)

# ......................................


class Rosenbrock(Function):
    def __init__(self, seed, name):
        orig_bounds = [-5.0, 10.0]
        char = ["unimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):

        x0 = x[:-1]
        x1 = x[1:]
        c = sum((1 - x0) ** 2) + 100 * sum((x1 - x0**2) ** 2)
        return c


funclist.append(Rosenbrock)

# ......................................


class Schwefel(Function):
    def __init__(self, seed, name):
        orig_bounds = [-500.0, 500.0]
        char = ["multimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):
        n = len(x)

        c = 418.9829 * n - sum(x * np.sin(np.sqrt(abs(x))))
        return c


funclist.append(Schwefel)

# ......................................


class Rastrigin(Function):
    def __init__(self, seed, name):
        orig_bounds = [-5.12, 5.12]
        char = ["multimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):
        n = len(x)
        c = 10 * n + sum(x**2 - 10 * np.cos(2 * np.pi * x))
        return c


funclist.append(Rastrigin)

# ......................................
# class Easom(Function): #2D
#     def __init__(self,seed,name):
#         super().__init__(self,seed,name)
#         orig_bounds = [-100.,100.]
#         self.getx0()
#         self.char.append('steep')
#         self.char.append('unimodal')

#     def func(self, x):
#         x1 = x[:-1]
#         x2 = x[1:]
#         c = -np.cos(x1)*np.cos(x2)*np.exp(-(x1-np.pi)**2-(x2-np.pi)**2)
#         return c

# funclist.append(Easom)

# ......................................


class Styblinski(Function):
    def __init__(self, seed, name):
        orig_bounds = [-5.0, 5.0]
        char = ["multimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):

        c = 0.5 * sum(x**4 - 16 * x**2 + 5 * x)
        return c


funclist.append(Styblinski)

# ......................................


class Branin(Function):  # 2D
    def __init__(self, seed, name):
        orig_bounds = [0.0, 15.0]
        char = ["multimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):

        x1 = x[:-1] + 5.0  # correct for the uneven box
        x2 = x[1:]

        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        cc = sum(
            a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        )
        return cc


funclist.append(Branin)


# ......................................


class SchafferF6(Function):  # 2D
    def __init__(self, seed, name):
        orig_bounds = [-100.0, 100.0]
        char = ["steep", "multimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):

        x1 = x[:-1]
        x2 = x[1:]

        x1x2 = x1**2 + x2**2

        c = sum(0.5 + (np.sin(np.sqrt(x1x2)) ** 2 - 0.5) / ((1 + 0.001 * x1x2) ** 2))
        return c


funclist.append(SchafferF6)

# ......................................


class Beale(Function):  # 2D
    def __init__(self, seed, name):
        orig_bounds = [-4.5, 4.5]
        char = ["multimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):

        x1 = x[:-1]
        x2 = x[1:]

        c = sum(
            (1.5 - x1 + x1 * x2) ** 2
            + (2.25 - x1 + x1 * x2**2) ** 2
            + (2.625 - x1 + x1 * x2**3) ** 2
        )
        return c


funclist.append(Beale)

# ......................................


class AckleyNo2(Function):
    def __init__(self, seed, name):
        orig_bounds = [-4.0, 4.0]
        char = ["unimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):
        n = len(x)

        x1 = x[:-1]
        x2 = x[1:]
        cc = sum(-200 * np.exp(-0.2 * np.sqrt(x1**2 + x2**2)))
        return cc


funclist.append(AckleyNo2)

# ......................................


class Bohachevsky(Function):
    def __init__(self, seed, name):
        orig_bounds = [-100.0, 100.0]
        char = ["unimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):
        n = len(x)

        x1 = x[:-1]
        x2 = x[1:]
        cc = sum(
            x1**2
            + 2 * x2**2
            - 0.3 * np.cos(3 * np.pi * x1)
            - 0.4 * np.cos(4 * np.pi * x2)
            + 0.7
        )
        return cc


funclist.append(Bohachevsky)

# ......................................


class Matyas(Function):
    def __init__(self, seed, name):
        orig_bounds = [-10.0, 10.0]
        char = ["unimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):
        n = len(x)

        x1 = x[:-1]
        x2 = x[1:]
        cc = sum(0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2)
        return cc


funclist.append(Matyas)

# ......................................


class Zakharov(Function):
    def __init__(self, seed, name):
        orig_bounds = [-5.0, 10.0]
        char = ["unimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):
        n = len(x)
        cc = (
            sum(x**2)
            + sum(0.5 * np.arange(1, len(x) + 1) * x) ** 2
            + sum(0.5 * np.arange(1, len(x) + 1) * x) ** 4
        )
        return cc


funclist.append(Zakharov)

# ......................................


class McCormick(Function):
    def __init__(self, seed, name):
        orig_bounds = [-3.0, 4]
        char = ["unimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):
        n = len(x)
        x1 = x[:-1]
        x2 = x[1:]

        cc = sum(np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1)
        return cc


funclist.append(McCormick)

# ......................................


class Leon(Function):
    def __init__(self, seed, name):
        orig_bounds = [-5.0, 5.0]
        char = ["unimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):
        n = len(x)
        x1 = x[:-1]
        x2 = x[1:]

        cc = sum(100 * (x2 - x1**3) ** 2 + (1 - x1) ** 2)
        return cc


funclist.append(Leon)

# ......................................

# CEC2013 Test functions are only defined for dimensions 2,5,10,20,30,40,50,60,70,80,90,100
class CEC2013(Function):
    def __init__(self, seed, name, number=None):

        if number == None:
            number = np.random.randint(1, 29)

        self.pgprob = pg.problem(pg.cec2013(prob_id=number, dim=DIMENSIONALITY))
        orig_bounds = [
            self.pgprob.get_bounds()[0][0],
            self.pgprob.get_bounds()[1][0],
        ]

        if 1 <= number <= 5:
            char = ["unimodal"]
        if 6 <= number <= 29:
            char = ["multimodal"]
            
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):
        c = self.pgprob.fitness(x)[0]
        return c


funclist.extend([CEC2013] * 10)  # 10 times!

# ......................................


class LinearRegression(Function):  # 4D, minimal 2D!
    # learning to opt benchmark!
    def __init__(self, seed, name):
        np.random.seed(seed=seed)
        d = 3  # self.dim-1
        _means = np.random.random(size=(4, d))
        _bias = np.random.random(4)

        self._xi, self._yi = [], []
        for i in range(25):
            choice = np.random.randint(4)
            xx = np.random.multivariate_normal(_means[choice], np.identity(d))
            yy = (
                (np.dot(xx, _means[choice]) / np.linalg.norm(_means[choice]))
                + _bias[choice]
                + np.random.normal()
            )

            self._xi.append(xx)
            self._yi.append(yy)

        self._xi = np.array(self._xi)
        self._yi = np.array(self._yi)

        super().__init__(seed, name, dim=4)
        self.sigma = 0
        self.m = np.identity(self.dim)
        self.o = np.zeros(self.dim)
        orig_bounds = [-10, 10]
        # self.getx0()
        self.sigma = 0

    def func(self, x):

        ww = x[:-1]
        bb = x[-1]
        c = 1
        d = sum(
            ((self._yi - np.inner(ww, self._xi) - bb) ** 2)
            / (c**2 + (self._yi - np.inner(ww, self._xi) - bb) ** 2)
        ) * (1 / 25)

        if np.isinf(d) or np.isnan(d):
            d = 10e5
        return d


# funclist.append(LinearRegression)

# ......................................


class LogisticRegression(Function):  # 4D, minimal 2D!
    # learning to opt benchmark!
    def __init__(self, seed, name):
        np.random.seed(seed=seed)
        d = 3  # self.dim-1
        _means = np.random.random(size=(2, d))
        _cov = np.random.random((d, d, 2))
        _labels = np.array([0, 1])

        self._xi, self._yi = [], []
        xx1 = np.random.multivariate_normal(
            _means[0], np.dot(_cov[:, :, 0], _cov[:, :, 0].T), size=50
        )
        xx2 = np.random.multivariate_normal(
            _means[1], np.dot(_cov[:, :, 1], _cov[:, :, 1].T), size=50
        )

        self._xi = np.r_[xx1, xx2]
        self._yi = np.array([0] * 50 + [1] * 50)

        super().__init__(self, seed, name, dim=4)
        self.sigma = 0
        self.m = np.identity(self.dim)
        self.o = np.zeros(self.dim)
        orig_bounds = [-5.0, 5.0]
        # self.getx0()
        self.sigma = 0

    def s_func(self, z):
        return 1 / (1 + np.exp(-z))

    def func(self, x):
        l = 0.0005
        ww = x[:-1]
        bb = x[-1]
        c = (
            -1
            / 100
            * sum(
                self._yi * np.log10(self.s_func(np.inner(ww, self._xi) + bb))
                + (1 - self._yi)
                * np.log10(1 - self.s_func(np.inner(ww, self._xi) + bb))
                + l / 2 * (np.sqrt(ww.dot(ww))) ** 2
            )
        )

        if np.isinf(c) or np.isnan(c):
            c = 10e5
        return c


# funclist.append(LogisticRegression)

# ......................................


if __name__ == "__main__":
    "This chooses a function based on a seed and plots the response surface"
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)



