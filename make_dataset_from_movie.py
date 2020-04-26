import numpy as np
import random

def loadMovieData():

    CUBE_FILENAMES = ["/Users/adamyedidia/lux/movies/15ps-blur-nn-cube-1.npy",
        "/Users/adamyedidia/lux/movies/15ps-blur-nn-cube-2.npy",
        "/Users/adamyedidia/lux/movies/15ps-blur-nn-cube-3.npy",
        "/Users/adamyedidia/lux/movies/15ps-blur-nn-cube-4.npy"]

    SPHERE_FILENAMES = ["/Users/adamyedidia/lux/movies/15ps-blur-nn-sphere-fine-1.npy",
        "/Users/adamyedidia/lux/movies/15ps-blur-nn-sphere-fine-2.npy",
        "/Users/adamyedidia/lux/movies/15ps-blur-nn-sphere-fine-3.npy",
        "/Users/adamyedidia/lux/movies/15ps-blur-nn-sphere-fine-4.npy"]

    cubeMovieArray = [np.load(movie) for movie in CUBE_FILENAMES]
    sphereMovieArray = [np.load(movie) for movie in SPHERE_FILENAMES]

    increaseFactor = 1e12

    dataSet = []

    for movie in cubeMovieArray:
        for grid32x32 in movie:
            if grid32x32.any() > 0:
                dataSet.append((np.reshape(grid32x32.flatten()*increaseFactor, \
                    (1024, 1)), np.reshape(np.array([1,0]), (2, 1))))

    for movie in sphereMovieArray:
        for grid32x32 in movie:
            if grid32x32.any() > 0:
                dataSet.append((np.reshape(grid32x32.flatten()*increaseFactor, \
                    (1024, 1)), np.reshape(np.array([0,1]), (2, 1))))

    random.shuffle(dataSet)

    n = len(dataSet)
    TRAINING_SET_FRACTION = 0.75
    VALIDATION_SET_FRACTION = 0.
    TEST_SET_FRACTION = 0.25

    assert TRAINING_SET_FRACTION + VALIDATION_SET_FRACTION + \
        TEST_SET_FRACTION == 1.

    return dataSet[:int(n*TRAINING_SET_FRACTION)], \
        dataSet[int(n*TRAINING_SET_FRACTION):int(n*(TRAINING_SET_FRACTION+VALIDATION_SET_FRACTION))], \
        dataSet[int(n*(TRAINING_SET_FRACTION+VALIDATION_SET_FRACTION)):]

def loadJustOneMovie(movieIndex):

        CUBE_FILENAMES = ["/Users/adamyedidia/lux/movies/15ps-blur-nn-cube-1.npy",
            "/Users/adamyedidia/lux/movies/15ps-blur-nn-cube-2.npy",
            "/Users/adamyedidia/lux/movies/15ps-blur-nn-cube-3.npy",
            "/Users/adamyedidia/lux/movies/15ps-blur-nn-cube-4.npy"]

        SPHERE_FILENAMES = ["/Users/adamyedidia/lux/movies/15ps-blur-nn-sphere-fine-1.npy",
            "/Users/adamyedidia/lux/movies/15ps-blur-nn-sphere-fine-2.npy",
            "/Users/adamyedidia/lux/movies/15ps-blur-nn-sphere-fine-3.npy",
            "/Users/adamyedidia/lux/movies/15ps-blur-nn-sphere-fine-4.npy"]

        if movieIndex < 4 and movieIndex >= 0:
            usedCubeMovies = [CUBE_FILENAMES[movieIndex]]
            usedSphereMovies = []
        if movieIndex < 8 and movieIndex >= 4:
            usedCubeMovies = []
            usedSphereMovies = [SPHERE_FILENAMES[movieIndex - 4]]

        cubeMovieArray = [np.load(movie) for movie in usedCubeMovies]
        sphereMovieArray = [np.load(movie) for movie in usedSphereMovies]

        increaseFactor = 1e12

        dataSet = []

        for movie in cubeMovieArray:
            for grid32x32 in movie:
                if grid32x32.any() > 0:
                    dataSet.append((np.reshape(grid32x32.flatten()*increaseFactor, \
                        (1024, 1)), np.reshape(np.array([1,0]), (2, 1))))

        for movie in sphereMovieArray:
            for grid32x32 in movie:
                if grid32x32.any() > 0:
                    dataSet.append((np.reshape(grid32x32.flatten()*increaseFactor, \
                        (1024, 1)), np.reshape(np.array([0,1]), (2, 1))))

        random.shuffle(dataSet)

        n = len(dataSet)
        TRAINING_SET_FRACTION = 0.75
        VALIDATION_SET_FRACTION = 0.
        TEST_SET_FRACTION = 0.25

        assert TRAINING_SET_FRACTION + VALIDATION_SET_FRACTION + \
            TEST_SET_FRACTION == 1.

        return dataSet[:int(n*TRAINING_SET_FRACTION)], \
            dataSet[int(n*TRAINING_SET_FRACTION):int(n*(TRAINING_SET_FRACTION+VALIDATION_SET_FRACTION))], \
            dataSet[int(n*(TRAINING_SET_FRACTION+VALIDATION_SET_FRACTION)):]
