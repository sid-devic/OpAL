# OpAL: Open Active Learning
From [wikipedia][1]:
"There are situations in which unlabeled data is abundant but manually labeling is expensive. In such a scenario, learning algorithms can actively query the user/teacher for labels. This type of iterative supervised learning is called active learning."

We found that most deep active learning is done only experimentally, and that there are no open source frameworks for groups
looking to actively use and deploy current SoTA active learning methods on deep models. OpAL is our attempt at remedying this problem.

Forked from [discriminative active learning](https://github.com/dsgissin/DiscriminativeActiveLearning), which implemented all the methods used.

[1]: https://en.wikipedia.org/wiki/Active_learning_(machine_learning)

## Dependencies

In order to run our code, you'll need these main packages:

- [Python](https://www.python.org/)>=3.5
- [Numpy](http://www.numpy.org/)>=1.14.3
- [Scipy](https://www.scipy.org/)>=1.0.0
- [TensorFlow](https://www.tensorflow.org/)>=1.5
- [Keras](https://keras.io/)>=2.2
- [Gurobi](http://www.gurobi.com/documentation/)>=8.0 (for the core set MIP query strategy)
- [Cleverhans](https://github.com/tensorflow/cleverhans)>=2.1 (for the adversarial query strategy)


### Possible Method Names
These are the possible names of methods that can be used in the experiments:
- "Random": random sampling
- "CoreSet": the greedy core set approach
- "CoreSetMIP": the core set with the MIP formulation
- "Uncertainty": uncertainty sampling with minimal top confidence
- "UncertaintyEntropy": uncertainty sampling with maximal entropy
- "Bayesian": Bayesian uncertainty sampling with minimal top confidence
- "BayesianEntropy": Bayesian uncertainty sampling with maximal entropy
- "EGL": estimated gradient length
- "Adversarial": adversarial active learning using DeepFool

### TODO
- Multi active-learning experiment support simultaneously
