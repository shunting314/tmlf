# tmlf
A simply machine learning framework.

The main purpose I start this project is to understand in deep how a machine learning framework does all the interesting stuff (espacially autograd).

This project relies on Eigen to represent tensor and do linear algebra.

Check nb/tmlf_nn.ipynb whichs achieves 96% accuracy on [Kaggle digit recognizer contest](https://www.kaggle.com/c/digit-recognizer).

## Instructions to build and run
Use tmlf/build script to build the tmlf_pybind.[PLATFORM_STRING].so . Make sure the root directory of the repository is under python search path. Then you can directly use tmlf package in your code like:
```python
from tmlf.python import workspace
```
