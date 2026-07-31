#include "../search/searchparams.h"

//Default search params
//The intent is that the are good default guesses for values of the parameters,
//with deterministic behavior (no noise, no randomization) and no bounds (unbounded time and visits).
//They are not necessarily the best parameters though, and have been kept mostly fixed over time even as things
//have changed to preserve the behavior of tests.
SearchParams::SearchParams()
  :winLossUtilityFactor(1.0),
   noResultUtilityForWhite(0.0),
   multiValueHeadUtilityMix(0.0),
   multiValueHeadSelectionBias(0.0),
   multiHeadNormalPolicyHead1Mix(0.0),
   multiHeadObjectiveSearchStrength(0.0),
   multiHeadObjectiveSelectionPower(2.0),
   multiHeadObjectiveSelectionSharpness(0.0),
   multiHeadObjectivePolicyMix(0.0),
   multiHeadObjectiveSeparatePlayouts(false),
   multiHeadObjectiveCrossWeight(0.0),
   multiHeadObjectiveValueWeightExponent(1.0),
   multiHeadObjectiveVctWeight(1.0),
   multiHeadVctObjectiveBudgetMix(1.0),
   multiHeadVctMaxAttackProp(0.0),
   multiHeadVctMaxDefenseProp(0.0),
   multiHeadVctUcbCoeff(1.0),
   multiHeadVctProbScale(0.20),
   multiHeadVctProbPower(1.0),
   multiHeadVctMainWinSuppression(0.0),
   multiHeadVctPolicyMix(1.0),
   multiHeadVctPolicyConsensusMix(0.0),
   multiHeadVctUseNormalRules(false),
   multiHeadVctProbeStartFraction(0.0),
   multiHeadVctProbeEndFraction(1.0),
   multiHeadVctProbeRampFraction(0.0),
   multiHeadVctPriorVisits(4.0),
   multiHeadVctNormalPolicyMaxMix(0.0),
   multiHeadVctAttackPolicyScale(1.0),
   multiHeadVctDefensePolicyScale(1.0),
   multiHeadVctPolicyRawProbMix(0.0),
   multiHeadVctNormalPolicyPow(2.0),
   multiHeadDrawWinNormalPolicyMaxMix(0.0),
   multiHeadDrawLossNormalPolicyMaxMix(0.0),
   multiHeadDrawNormalPolicyProbScale(0.20),
   multiHeadDrawPolicyRawProbMix(0.0),
   multiHeadDrawNormalPolicyPow(1.0),
   multiHeadDrawPolicyFlattening(0.0),
   multiHeadDrawRootMinVisitsCoeff(0.0),
   multiHeadDrawAuxRootVisits(0.0),
   multiHeadTacticalRootVisits(0.0),
   multiHeadTacticalContrastiveMix(0.0),
   multiHeadTacticalObjectiveMask(15),
   multiHeadTacticalDisproofStrength(0.0),
   multiHeadTacticalPolicyPower(1.0),
   multiHeadDrawForcedReplyRootVisits(0.0),
   multiHeadDrawForcedReplyTreeVisits(0.0),
   multiHeadDrawForcedReplyAuxPolicyGate(0.0),
   multiHeadDrawForcedReplySidecarVisits(0.0),
   multiHeadDrawForcedReplySidecarMaxProp(0.0),
   multiHeadDrawForcedReplyPolicyThreshold(0.5),
   multiHeadVctGuidedPlayoutProp(0.0),
   multiHeadDrawGuidedPlayoutProp(0.0),
   multiHeadGuidedStartFraction(0.0),
   multiHeadGuidedPolicyMix(1.0),
   multiHeadGuidedValueWeight(0.0),
   multiHeadOutcomeIntervalExplore(0.0),
   multiHeadOutcomeIntervalVisitScale(8.0),
   multiHeadAuxPolicyChildGate(0.0),
   multiHeadAuxPolicyVisitScale(0.0),
   multiHeadAuxPolicyOptimism(0.0),
   multiHeadAuxPolicyConcentrationScale(0.10),
   multiHeadAuxValueExplore(0.0),
   multiHeadAuxValueVisitScale(8.0),
   multiHeadVctCpuctScale(1.0),
   multiHeadVctValueWeightExponent(0.5),
   multiHeadVctValidationProp(0.25),
   multiHeadVctValidationUtility(2.0),
   multiHeadVctValidationOptimism(-1.0),
   multiHeadVctNormalVerificationProp(0.0),
   multiHeadVctMoveSelectionWeight(0.0),
   multiHeadVctRelativeMoveSelectionWeight(0.0),
   multiHeadVctNnMoveSelectionWeight(0.0),
   multiHeadVctMoveSelectionVisitScale(8.0),
   noResultUtilityReduce(0.0),
   cpuctExploration(1.0),
   cpuctExplorationLog(0.0),
   cpuctExplorationBase(500),
   cpuctUtilityStdevPrior(0.25),
   cpuctUtilityStdevPriorWeight(1.0),
   cpuctUtilityStdevScale(0.0),
   fpuReductionMax(0.2),
   fpuLossProp(0.0),
   fpuParentWeightByVisitedPolicy(false),
   fpuParentWeightByVisitedPolicyPow(1.0),
   fpuParentWeight(0.0),
   valueWeightExponent(0.5),
   useNoisePruning(false),
   noisePruneUtilityScale(0.15),
   noisePruningCap(1e50),
   useUncertainty(false),
   uncertaintyCoeff(0.2),
   uncertaintyExponent(1.0),
   uncertaintyMaxWeight(8.0),
   useGraphSearch(false),
   graphSearchCatchUpLeakProb(0.0),
   //graphSearchCatchUpProp(0.0),
   rootNoiseEnabled(false),
   rootDirichletNoiseTotalConcentration(10.83),
   rootDirichletNoiseWeight(0.25),
   rootPolicyTemperature(1.0),
   rootPolicyTemperatureEarly(1.0),
   rootFpuReductionMax(0.2),
   rootFpuLossProp(0.0),
   rootNumSymmetriesToSample(1),
   rootSymmetryPruning(false),
   rootDesiredPerChildVisitsCoeff(0.0),
   chosenMoveTemperature(0.0),
   chosenMoveTemperatureEarly(0.0),
   chosenMoveTemperatureHalflife(19),
   chosenMoveSubtract(0.0),
   chosenMovePrune(1.0),
   useLcbForSelection(false),
   lcbStdevs(4.0),
   minVisitPropForLCB(0.05),
   useNonBuggyLcb(false),
   rootPruneUselessMoves(false),
   wideRootNoise(0.0),
   playoutDoublingAdvantage(0.0),
   playoutDoublingAdvantagePla(C_EMPTY),
   avoidRepeatedPatternUtility(0.0),
   nnPolicyTemperature(1.0f),
   useVCFInput(true),
   useForbiddenInput(true),
   useHistoryInput(true),
   fourAttackPolicyReduce(0.0),
   subtreeValueBiasFactor(0.0),
   subtreeValueBiasTableNumShards(65536),
   subtreeValueBiasFreeProp(0.8),
   subtreeValueBiasWeightExponent(0.5),
   nodeTableShardsPowerOfTwo(16),
   numVirtualLossesPerThread(3.0),
   numThreads(1),   
   backendNumThreads(1),
   maxVisits(((int64_t)1) << 50),
   maxPlayouts(((int64_t)1) << 50),
   maxTime(1.0e20),
   maxVisitsPondering(((int64_t)1) << 50),
   maxPlayoutsPondering(((int64_t)1) << 50),
   maxTimePondering(1.0e20),
   lagBuffer(0.0),
   treeReuseCarryOverTimeFactor(0.0),
   overallocateTimeFactor(1.0),
   midgameTimeFactor(1.0),
   midgameTurnPeakTime(130.0),
   endgameTurnTimeDecay(100.0),
   obviousMovesTimeFactor(1.0),
   obviousMovesPolicyEntropyTolerance(0.30),
   obviousMovesPolicySurpriseTolerance(0.15),
   futileVisitsThreshold(0.0),
   finishGameSearchDelayMicroseconds(0)
{}

SearchParams::~SearchParams()
{}

SearchParams SearchParams::forTestsV1() {
  SearchParams params;
  params.cpuctExploration = 0.9;
  params.cpuctExplorationLog = 0.4;
  params.rootFpuReductionMax = 0.1;
  params.rootPolicyTemperatureEarly = 1.2;
  params.rootPolicyTemperature = 1.1;
  params.useLcbForSelection = true;
  params.lcbStdevs = 5;
  params.minVisitPropForLCB = 0.15;
  params.rootPruneUselessMoves = true;
  params.useNonBuggyLcb = true;
  return params;
}

SearchParams SearchParams::forTestsV2() {
  SearchParams params;
  params.cpuctExploration = 0.9;
  params.cpuctExplorationLog = 0.4;
  params.rootFpuReductionMax = 0.1;
  params.rootPolicyTemperatureEarly = 1.2;
  params.rootPolicyTemperature = 1.1;
  params.useLcbForSelection = true;
  params.lcbStdevs = 5;
  params.minVisitPropForLCB = 0.15;
  params.rootPruneUselessMoves = true;
  params.useNonBuggyLcb = true;
  params.useGraphSearch = true;
  params.fpuParentWeightByVisitedPolicy = true;
  params.valueWeightExponent = 0.25;
  params.useNoisePruning = true;
  params.useUncertainty = true;
  params.uncertaintyCoeff = 0.25;
  params.uncertaintyExponent = 1.0;
  params.uncertaintyMaxWeight = 8.0;
  params.cpuctUtilityStdevPrior = 0.40;
  params.cpuctUtilityStdevPriorWeight = 2.0;
  params.cpuctUtilityStdevScale = 0.85;
  params.subtreeValueBiasFactor = 0.45;
  params.subtreeValueBiasFreeProp = 0.8;
  params.subtreeValueBiasWeightExponent = 0.85;
  return params;
}

void SearchParams::failIfParamsDifferOnUnchangeableParameter(const SearchParams& initial, const SearchParams& dynamic) {
  if(dynamic.numThreads > initial.numThreads) {
    throw StringError("Cannot increase number of search threads after initialization since this is used to initialize neural net buffer capacity");
  }
  if(dynamic.nodeTableShardsPowerOfTwo != initial.nodeTableShardsPowerOfTwo) {
    throw StringError("Cannot change nodeTableShardsPowerOfTwo after initialization");
  }
}


#define PRINTPARAM(PARAMNAME) out << #PARAMNAME << ": " << PARAMNAME << std::endl;
void SearchParams::printParams(std::ostream& out) {


  PRINTPARAM(winLossUtilityFactor);
  PRINTPARAM(noResultUtilityForWhite);
  PRINTPARAM(multiValueHeadUtilityMix);
  PRINTPARAM(multiValueHeadSelectionBias);
  PRINTPARAM(multiHeadNormalPolicyHead1Mix);
  PRINTPARAM(multiHeadObjectiveSearchStrength);
  PRINTPARAM(multiHeadObjectiveSelectionPower);
  PRINTPARAM(multiHeadObjectiveSelectionSharpness);
  PRINTPARAM(multiHeadObjectivePolicyMix);
  PRINTPARAM(multiHeadObjectiveSeparatePlayouts);
  PRINTPARAM(multiHeadObjectiveCrossWeight);
  PRINTPARAM(multiHeadObjectiveValueWeightExponent);
  PRINTPARAM(multiHeadObjectiveVctWeight);
  PRINTPARAM(multiHeadVctObjectiveBudgetMix);
  PRINTPARAM(multiHeadVctMaxAttackProp);
  PRINTPARAM(multiHeadVctMaxDefenseProp);
  PRINTPARAM(multiHeadVctUcbCoeff);
  PRINTPARAM(multiHeadVctProbScale);
  PRINTPARAM(multiHeadVctProbPower);
  PRINTPARAM(multiHeadVctMainWinSuppression);
  PRINTPARAM(multiHeadVctPolicyMix);
  PRINTPARAM(multiHeadVctPolicyConsensusMix);
  PRINTPARAM(multiHeadVctUseNormalRules);
  PRINTPARAM(multiHeadVctProbeStartFraction);
  PRINTPARAM(multiHeadVctProbeEndFraction);
  PRINTPARAM(multiHeadVctProbeRampFraction);
  PRINTPARAM(multiHeadVctPriorVisits);
  PRINTPARAM(multiHeadVctNormalPolicyMaxMix);
  PRINTPARAM(multiHeadVctAttackPolicyScale);
  PRINTPARAM(multiHeadVctDefensePolicyScale);
  PRINTPARAM(multiHeadVctPolicyRawProbMix);
  PRINTPARAM(multiHeadVctNormalPolicyPow);
  PRINTPARAM(multiHeadDrawWinNormalPolicyMaxMix);
  PRINTPARAM(multiHeadDrawLossNormalPolicyMaxMix);
  PRINTPARAM(multiHeadDrawNormalPolicyProbScale);
  PRINTPARAM(multiHeadDrawPolicyRawProbMix);
  PRINTPARAM(multiHeadDrawNormalPolicyPow);
  PRINTPARAM(multiHeadDrawPolicyFlattening);
  PRINTPARAM(multiHeadDrawRootMinVisitsCoeff);
  PRINTPARAM(multiHeadDrawAuxRootVisits);
  PRINTPARAM(multiHeadTacticalRootVisits);
  PRINTPARAM(multiHeadTacticalContrastiveMix);
  PRINTPARAM(multiHeadTacticalObjectiveMask);
  PRINTPARAM(multiHeadTacticalDisproofStrength);
  PRINTPARAM(multiHeadTacticalPolicyPower);
  PRINTPARAM(multiHeadDrawForcedReplyRootVisits);
  PRINTPARAM(multiHeadDrawForcedReplyTreeVisits);
  PRINTPARAM(multiHeadDrawForcedReplyAuxPolicyGate);
  PRINTPARAM(multiHeadDrawForcedReplySidecarVisits);
  PRINTPARAM(multiHeadDrawForcedReplySidecarMaxProp);
  PRINTPARAM(multiHeadDrawForcedReplyPolicyThreshold);
  PRINTPARAM(multiHeadVctGuidedPlayoutProp);
  PRINTPARAM(multiHeadDrawGuidedPlayoutProp);
  PRINTPARAM(multiHeadGuidedStartFraction);
  PRINTPARAM(multiHeadGuidedPolicyMix);
  PRINTPARAM(multiHeadGuidedValueWeight);
  PRINTPARAM(multiHeadOutcomeIntervalExplore);
  PRINTPARAM(multiHeadOutcomeIntervalVisitScale);
  PRINTPARAM(multiHeadAuxPolicyChildGate);
  PRINTPARAM(multiHeadAuxPolicyVisitScale);
  PRINTPARAM(multiHeadAuxPolicyOptimism);
  PRINTPARAM(multiHeadAuxPolicyConcentrationScale);
  PRINTPARAM(multiHeadAuxValueExplore);
  PRINTPARAM(multiHeadAuxValueVisitScale);
  PRINTPARAM(multiHeadVctCpuctScale);
  PRINTPARAM(multiHeadVctValueWeightExponent);
  PRINTPARAM(multiHeadVctValidationProp);
  PRINTPARAM(multiHeadVctValidationUtility);
  PRINTPARAM(multiHeadVctValidationOptimism);
  PRINTPARAM(multiHeadVctNormalVerificationProp);
  PRINTPARAM(multiHeadVctMoveSelectionWeight);
  PRINTPARAM(multiHeadVctRelativeMoveSelectionWeight);
  PRINTPARAM(multiHeadVctNnMoveSelectionWeight);
  PRINTPARAM(multiHeadVctMoveSelectionVisitScale);

  PRINTPARAM(cpuctExploration);
  PRINTPARAM(cpuctExplorationLog);
  PRINTPARAM(cpuctExplorationBase);

  PRINTPARAM(cpuctUtilityStdevPrior);
  PRINTPARAM(cpuctUtilityStdevPriorWeight);
  PRINTPARAM(cpuctUtilityStdevScale);

  PRINTPARAM(fpuReductionMax);
  PRINTPARAM(fpuLossProp);

  PRINTPARAM(fpuParentWeightByVisitedPolicy);
  PRINTPARAM(fpuParentWeightByVisitedPolicyPow);
  PRINTPARAM(fpuParentWeight);


  PRINTPARAM(valueWeightExponent);
  PRINTPARAM(useNoisePruning);
  PRINTPARAM(noisePruneUtilityScale);
  PRINTPARAM(noisePruningCap);


  PRINTPARAM(useUncertainty);
  PRINTPARAM(uncertaintyCoeff);
  PRINTPARAM(uncertaintyExponent);
  PRINTPARAM(uncertaintyMaxWeight);


  PRINTPARAM(useGraphSearch);
  PRINTPARAM(graphSearchCatchUpLeakProb);



  PRINTPARAM(rootNoiseEnabled);
  PRINTPARAM(rootDirichletNoiseTotalConcentration);
  PRINTPARAM(rootDirichletNoiseWeight);

  PRINTPARAM(rootPolicyTemperature);
  PRINTPARAM(rootPolicyTemperatureEarly);
  PRINTPARAM(rootFpuReductionMax);
  PRINTPARAM(rootFpuLossProp);
  PRINTPARAM(rootNumSymmetriesToSample);
  PRINTPARAM(rootSymmetryPruning);

  PRINTPARAM(rootDesiredPerChildVisitsCoeff);


  PRINTPARAM(chosenMoveTemperature);
  PRINTPARAM(chosenMoveTemperatureEarly);
  PRINTPARAM(chosenMoveTemperatureHalflife);
  PRINTPARAM(chosenMoveSubtract);
  PRINTPARAM(chosenMovePrune);

  PRINTPARAM(useLcbForSelection);
  PRINTPARAM(lcbStdevs);
  PRINTPARAM(minVisitPropForLCB);
  PRINTPARAM(useNonBuggyLcb);


  PRINTPARAM(rootPruneUselessMoves);
  PRINTPARAM(wideRootNoise);

  PRINTPARAM(playoutDoublingAdvantage);
  std::cout << "playoutDoublingAdvantagePla" << ": " << (int)playoutDoublingAdvantagePla << std::endl;

  PRINTPARAM(avoidRepeatedPatternUtility);

  PRINTPARAM(nnPolicyTemperature);

  PRINTPARAM(subtreeValueBiasFactor);
  PRINTPARAM(subtreeValueBiasTableNumShards);
  PRINTPARAM(subtreeValueBiasFreeProp);
  PRINTPARAM(subtreeValueBiasWeightExponent);


  PRINTPARAM(nodeTableShardsPowerOfTwo);
  PRINTPARAM(numVirtualLossesPerThread);


  PRINTPARAM(numThreads);
  PRINTPARAM(backendNumThreads);
  PRINTPARAM(maxVisits);
  PRINTPARAM(maxPlayouts);
  PRINTPARAM(maxTime);


  PRINTPARAM(maxVisitsPondering);
  PRINTPARAM(maxPlayoutsPondering);
  PRINTPARAM(maxTimePondering);


  PRINTPARAM(lagBuffer);




  PRINTPARAM(treeReuseCarryOverTimeFactor);
  PRINTPARAM(overallocateTimeFactor);
  PRINTPARAM(midgameTimeFactor);
  PRINTPARAM(midgameTurnPeakTime);
  PRINTPARAM(endgameTurnTimeDecay);
  PRINTPARAM(obviousMovesTimeFactor);
  PRINTPARAM(obviousMovesPolicyEntropyTolerance);
  PRINTPARAM(obviousMovesPolicySurpriseTolerance);

  PRINTPARAM(futileVisitsThreshold);
}
