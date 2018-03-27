function passrate = TestSuite()
  TestComputeAccuracy();
  TestComputeCost();
  TestEvaluateClassifier();
  TestGradientCalculations();
  TestLoadBatch();
  TestMiniBatchGD();
endfunction
