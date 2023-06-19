# Porous_PINNs

Source code and files for building a Physics Informed Neural Network surrogate in PyTorch.

The surrogate is developed for extremely high-dimensional two-dimensional multi-phase porous flow simulations in OpenFoam. The surrogate is trained on a dataset of hundreds of millions of training instances and applied to extremely high numbers of multi-step-ahead forecasts:

Initial condition (t=0)      -> surrogate -> Pred state of system (t=1);
Pred state of system (t=1)   -> surrogate -> Pred state of system (t=2);
...
Pred state of system (t=N-1) -> surrogate -> Pred state of system (t=N);

The trained model is utilised through `inference.py` with the `args` specifying usage. 

#workinprogress#
