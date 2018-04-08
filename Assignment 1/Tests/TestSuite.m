## This file represent a test-suite that runs, in succession, tests to verify
## that dimensionalities and other defined expectations are in order.
## Currently, it seems only 'TestGradientCalculations' fail, where 'fail' implies
## that our gradient in W has a maximum absolute valued relative difference between
## numerical and analytical gradient that varies between 0.01-1. Included in the repo
## is copies of the slides describing the gradient definitions, to which we adhere perfectly.
## the only thing to notice is that they dont transpose their addition of g(*) to the gradient
## of b, but it can be easily checked that dim(g()) == (1,K) whereas dim(grad_b) == (K,1).
## We therefore need that transpose to ensure that dimensionalities are correct.

function passrate = TestSuite()
  TestComputeAccuracy(); 
  TestComputeCost();
  TestEvaluateClassifier();
  TestGradientCalculations();
  TestLoadBatch();
  TestMiniBatchGD();
endfunction
