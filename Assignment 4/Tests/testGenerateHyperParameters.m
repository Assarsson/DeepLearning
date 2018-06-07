
%!test
%! addpath Functions/
%! hp = GenerateHyperParameters();
%! assert(hp.eta == 0.1);
%! assert(hp.m == 100);
%! assert(hp.seqLength == 25);
%! hp = GenerateHyperParameters(20);
%! assert(hp.m == 20);
%! assert(hp.eta == 0.1);
%! assert(hp.seqLength == 25);
%! hp = GenerateHyperParameters(30, 1000);
%! assert(hp.eta == 1000);
%! assert(hp.m == 30);
%! assert(hp.seqLength == 25);
%! hp = GenerateHyperParameters(40, 1000, 100);
%! assert(hp.eta == 1000);
%! assert(hp.m == 40);
%! assert(hp.seqLength == 100);
