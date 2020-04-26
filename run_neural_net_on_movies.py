from mnist_loader import load_data_wrapper
from make_dataset_from_movie import loadMovieData, loadJustOneMovie
from network_with_halfway_sensible_formatting import Network
#from network import Network

training_data, validation_data, test_data = loadMovieData()
#training_data, validation_data, test_data = load_data_wrapper()
print len(test_data)
print len(test_data[0])
print test_data[0][0].shape
print test_data[0][1].shape

net = Network([1024, 30, 2])
#net = Network([784, 30, 10])

print net.SGD(training_data, 1, 10, 0.01, test_data=test_data)
for i in range(8):
    print net.evaluate(loadJustOneMovie(i)[2])
