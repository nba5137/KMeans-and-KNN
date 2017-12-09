# CSCI 3302: Homework 3 -- Clustering and Classification
# Implementations of K-Means clustering and K-Nearest Neighbor classification

# Sibo Song

###########################################################################
## The pkl file looks weird.. I think it's because I use numpy.zeros... ###
###########################################################################

import pickle
import numpy as np
import random
import copy

test = []

class KMeansClassifier(object):

  def __init__(self):
    self._cluster_centers = [] # List of points representing cluster centers
    self._data = [] # List of datapoints (lists of real values)

  def add_datapoint(self, datapoint):
    self._data.append(datapoint)

  def fit(self, k):
    # Fit k clusters to the data, by starting with k randomly selected cluster centers.
    # HINT: To choose reasonable initial cluster centers, you can set them to be in the same spot as random (different) points from the dataset
    self._cluster_centers = []
    
    '''
    ##Got help from Prof. Hayes. Getting the center of whole graph.
    my_list = [0]*len(self._data[0])
    for i in range(len(self._data)):
      for j in range(len(self._data[i])):
        my_list[j] += self._data[i][j]
        
    for i in range(len(my_list)):
      my_list[i] /= len(self._data)    

    ## Going to choose k points around the center of whole 1graph. ~[0.3,1.0]
    p = np.zeros((k, len(self._data[0])))
    for i in range(k):
      p[i] = [my_list[0] + 0.01 * random.randint(0, 20), my_list[1] + 0.1 * random.randint(0, 10)]
      self._cluster_centers.append(p[i])

    for i in range(k):
      for j in range(k):
        if (p[i][0] == p[j][0]):
          for l in range(len(p[0])):
            p[j][l] += 0.22 # New 0 division proof
    '''
    ################################################################
    # Got help from Prof. Hayes. Choose k indices at random from between 0 and len(self._data)
    random_indices_list = random.sample(range(len(self._data)), k)

    # Got help from Prof. Hayes. Add those k indices to the "cluster_centers"" list
    for cluster_center_idx in random_indices_list: 
        self._cluster_centers.append(copy.copy(self._data[cluster_center_idx]))
    
    # TODO Follow convergence procedure to find final locations for each center
    def dist(p1, p2):  
     distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)  
     return distance  

    ## Setting number of movement. Thanks to Prof. Hayes. just understand what K-means is finally!
    cluster_center_movement = 100 
    threshold = 1e-3 

    while cluster_center_movement > threshold: ## iterate to recalc new centers.
    
      ## Depart all points into k groups with labels (Classify?)
      ##labels for groups
      label = [0]*(len(self._data)) ##arrange with 0,1,2...k-1
      
      for i in range(len(self._data)):
        dist_old = 999999 # Reset for each
        for j in range(len(self._cluster_centers)):
          dist_new = dist(self._data[i], self._cluster_centers[j])
          if (dist_new < dist_old):
            dist_old = dist_new
            label[i] = j ## Record the label for each point

      ## Once there are k groups of points, recalc for centers
      p_n = np.zeros((k, len(self._data[0])))
      
      p_s = [0]*len(self._cluster_centers) #size of groups

    
      for i in range(len(self._data)):
        for j in range(len(self._cluster_centers)):
          for h in range(len(self._data[0])):
            ## Give points to each cluster
            if(label[i] == j):
              p_n[j][h] += self._data[i][h]
              p_s[j] += 1

      ## Calc for new centers. Finished verison :)
      for i in range(len(p_n)):
        for j in range(len(p_n[0])):
          if (p_s[i] != 0):
            p_n[i][j] = p_n[i][j]/((p_s[i])) # Still need this, since a cluster center 'double nan/0' occurs sometime.
          else : ## Tring to get it pretty, actually it is a double Nan/0
            p_n[i][j] = p_n[i][j]/((p_s[i] + 0.1)) + random.uniform(my_list[j]/2,my_list[j]*2)

      # TODO Add each of the 'k' final cluster_centers to the model (self._cluster_centers)
      ## I update self._cluster_centers[] each time.
      for i in range(len(self._cluster_centers)):
        self._cluster_centers[i] = [p_n[i][0],p_n[i][1]]
      ###
      # This way will make the output look nice, but the pkl file looks weird.. I think it's because I use numpy.zeros...
      ###########
      cluster_center_movement -= 1 ## every iteration -1
      
  #################################################################

  def classify(self,p):
    # Given a data point p, figure out which cluster it belongs to and return that cluster's ID (its index in self._cluster_centers)
    closest_cluster_index = None

    # TODO Find nearest cluster center, then return its index in self._cluster_centers
    '''
    ## I suppose the p is a given index of self._data, and the id returned is the index of self_.cluster_centers[] like 0, 1, 2 ..
    # if p is like [1.0, 2.0]
    for i in range(len(self._data)):
      if (self._data[i] == p):
        closest_cluster_index = label[i] 
    
    ## if p is an index
    closest_cluster_index = label[p]
    '''
    # Since p is a new point, recalc dist between the new point and cluster centers
    for i in range(len(self._cluster_centers)):
      dist_new = dist(p, self._cluster_centers[i])
      if (dist_new < dist_old):
        dist_old = dist_new
        closest_cluster_index = i ## Record the label  

    return closest_cluster_index

class KNNClassifier(object):

  def __init__(self):
    self._data = [] # list of (data, label) tuples
  
  def clear_data(self):
    self._data = []

  def add_labeled_datapoint(self, data_point, label):
    self._data.append((data_point, label))
  

  def classify_datapoint(self, data_point, k):
    #label_counts = {} # Dictionary mapping "label" => count
    ## Since I don't find a specific tutorial about dict, I use another way with list.
    label_counts = []
    best_label = None

    #TODO: Perform k_nearest_neighbor classification, set best_label to majority label for k-nearest points
    alldist = [0] * (len(self._data))
    olddist = [999999] * k

    ##Added distance function for 2D
    def dist(p1, p2):
      distance = np.sqrt((p2[0] - p1[0][0])**2 + (p2[1] - p1[0][1])**2)  
      return distance

    #test = self._data
    ## ([a,b],'c')

    for i in range(len(self._data)):
      alldist[i] = dist(self._data[i], data_point) ## calculate all distance at first.

    ## Getting k shortest distances
    for i in range(len(alldist)):
      for j in range(k):
        if (alldist[i] < olddist[j]):
          olddist[j] = alldist[i]
          #label_counts.append(self._data[i][1])

    ## if we append a shortest point's label, it cannot be repeated.
    ## Appending labels for k points that have shortest distances to data point
    for i in range(len(alldist)):
      for j in range(k):
        if (alldist[i] == olddist[j]):
          label_counts.append(self._data[i][1])
    
    ## count for labels
    ## Getting all types of labels
    types = [label_counts[0]]
    for i in range(len(label_counts) - 1):
      if (label_counts[i] != label_counts[i+1]):
        types.append(label_counts[i+1])
    ## Counting
    numbers = [0]* (len(types))
    for i in range(len(label_counts)):
      for j in range(len(numbers)):
        if (types[j] == label_counts[i]):
          numbers[j] += 1

    major = 0
    for i in range(len(numbers)):
      if (numbers[i] > major):
        major = numbers[i]
        best_label = types[i]

    #test.append(best_label)
    return best_label



def print_and_save_cluster_centers(classifier, filename):
  for idx, center in enumerate(classifier._cluster_centers):
    print "  Cluster %d, center at: %s" % (idx, str(center))


  f = open(filename,'w')
  pickle.dump(classifier._cluster_centers, f)
  f.close()

def read_data_file(filename):
  f = open(filename)
  data_dict = pickle.load(f)
  f.close()

  return data_dict['data'], data_dict['labels']


def main():
  # read data file
  data, labels = read_data_file('hw3_data.pkl')

  # data is an 'N' x 'M' matrix, where N=number of examples and M=number of dimensions per example
  # data[0] retrieves the 0th example, a list with 'M' elements
  # labels is an 'N'-element list, where labels[0] is the label for the datapoint at data[0]


  ########## PART 1 ############
  # perform K-means clustering
  kMeans_classifier = KMeansClassifier()
  for datapoint in data:
    kMeans_classifier.add_datapoint(datapoint) # add data to the model

  kMeans_classifier.fit(4) # Fit 4 clusters to the data

  # plot results
  print '\n'*2
  print "K-means Classifier Test"
  print '-'*40
  print "Cluster center locations:"
  print_and_save_cluster_centers(kMeans_classifier, "hw3_kmeans_Sibo.pkl")


  print '\n'*2


  ########## PART 2 ############
  print "K-Nearest Neighbor Classifier Test"
  print '-'*40

  # Create and test K-nearest neighbor classifier
  kNN_classifier = KNNClassifier()
  k = 2

  correct_classifications = 0
  # Perform leave-one-out cross validation (LOOCV) to evaluate KNN performance
  for holdout_idx in range(len(data)):
    # Reset classifier
    kNN_classifier.clear_data()

    for idx in range(len(data)):
      if idx == holdout_idx: continue # Skip held-out data point being classified

      # Add (data point, label) tuples to KNNClassifier
      kNN_classifier.add_labeled_datapoint(data[idx], labels[idx])

    guess = kNN_classifier.classify_datapoint(data[holdout_idx], k) # Perform kNN classification
    if guess == labels[holdout_idx]: 
      correct_classifications += 1.0
  
  print "kNN classifier for k=%d" % k
  print "Accuracy: %g" % (correct_classifications / len(data))
  print '\n'*2
  #print "Additional printing...\n"
  #print "For now test is disntace index.\n" 
  #print (test)
  

if __name__ == '__main__':
  main()
