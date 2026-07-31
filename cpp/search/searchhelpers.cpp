#include "../search/search.h"

#include "../core/fancymath.h"
#include "../search/searchnode.h"
#include "../search/patternbonustable.h"

//------------------------
#include "../core/using.h"
//------------------------

uint32_t Search::chooseIndexWithTemperature(Rand& rand, const double* relativeProbs, int numRelativeProbs, double temperature) {
  assert(numRelativeProbs > 0);
  assert(numRelativeProbs <= Board::MAX_ARR_SIZE); //We're just doing this on the stack
  double processedRelProbs[Board::MAX_ARR_SIZE];

  double maxValue = 0.0;
  for(int i = 0; i<numRelativeProbs; i++) {
    if(relativeProbs[i] > maxValue)
      maxValue = relativeProbs[i];
  }
  assert(maxValue > 0.0);

  //Temperature so close to 0 that we just calculate the max directly
  if(temperature <= 1.0e-4) {
    double bestProb = relativeProbs[0];
    int bestIdx = 0;
    for(int i = 1; i<numRelativeProbs; i++) {
      if(relativeProbs[i] > bestProb) {
        bestProb = relativeProbs[i];
        bestIdx = i;
      }
    }
    return bestIdx;
  }
  //Actual temperature
  else {
    double logMaxValue = log(maxValue);
    double sum = 0.0;
    for(int i = 0; i<numRelativeProbs; i++) {
      //Numerically stable way to raise to power and normalize
      processedRelProbs[i] = relativeProbs[i] <= 0.0 ? 0.0 : exp((log(relativeProbs[i]) - logMaxValue) / temperature);
      sum += processedRelProbs[i];
    }
    assert(sum > 0.0);
    uint32_t idxChosen = rand.nextUInt(processedRelProbs,numRelativeProbs);
    return idxChosen;
  }
}

void Search::computeDirichletAlphaDistribution(int policySize, const float* policyProbs, double* alphaDistr) {
  int legalCount = 0;
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0)
      legalCount += 1;
  }

  if(legalCount <= 0)
    throw StringError("computeDirichletAlphaDistribution: No move with nonnegative policy value - can't even pass?");

  //We're going to generate a gamma draw on each move with alphas that sum up to searchParams.rootDirichletNoiseTotalConcentration.
  //Half of the alpha weight are uniform.
  //The other half are shaped based on the log of the existing policy.
  double logPolicySum = 0.0;
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0) {
      alphaDistr[i] = log(std::min(0.01, (double)policyProbs[i]) + 1e-20);
      logPolicySum += alphaDistr[i];
    }
  }
  double logPolicyMean = logPolicySum / legalCount;
  double alphaPropSum = 0.0;
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0) {
      alphaDistr[i] = std::max(0.0, alphaDistr[i] - logPolicyMean);
      alphaPropSum += alphaDistr[i];
    }
  }
  double uniformProb = 1.0 / legalCount;
  if(alphaPropSum <= 0.0) {
    for(int i = 0; i<policySize; i++) {
      if(policyProbs[i] >= 0)
        alphaDistr[i] = uniformProb;
    }
  }
  else {
    for(int i = 0; i<policySize; i++) {
      if(policyProbs[i] >= 0)
        alphaDistr[i] = 0.5 * (alphaDistr[i] / alphaPropSum + uniformProb);
    }
  }
}

void Search::addDirichletNoise(const SearchParams& searchParams, Rand& rand, int policySize, float* policyProbs) {
  double r[NNPos::MAX_NN_POLICY_SIZE];
  Search::computeDirichletAlphaDistribution(policySize, policyProbs, r);

  //r now contains the proportions with which we would like to split the alpha
  //The total of the alphas is searchParams.rootDirichletNoiseTotalConcentration
  //Generate gamma draw on each move
  double rSum = 0.0;
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0) {
      r[i] = rand.nextGamma(r[i] * searchParams.rootDirichletNoiseTotalConcentration);
      rSum += r[i];
    }
    else
      r[i] = 0.0;
  }

  //Normalized gamma draws -> dirichlet noise
  for(int i = 0; i<policySize; i++)
    r[i] /= rSum;

  //At this point, r[i] contains a dirichlet distribution draw, so add it into the nnOutput.
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0) {
      double weight = searchParams.rootDirichletNoiseWeight;
      policyProbs[i] = (float)(r[i] * weight + policyProbs[i] * (1.0-weight));
    }
  }
}


std::shared_ptr<NNOutput>* Search::maybeAddPolicyNoiseAndTemp(SearchThread& thread, bool isRoot, NNOutput* oldNNOutput) const {
  if(!isRoot)
    return NULL;
  if(!searchParams.rootNoiseEnabled && searchParams.rootPolicyTemperature == 1.0 && searchParams.rootPolicyTemperatureEarly == 1.0 && rootHintLoc == Board::NULL_LOC)
    return NULL;
  if(oldNNOutput == NULL)
    return NULL;
  if(oldNNOutput->noisedPolicyProbs != NULL)
    return NULL;

  //Copy nnOutput as we're about to modify its policy to add noise or temperature
  std::shared_ptr<NNOutput>* newNNOutputSharedPtr = new std::shared_ptr<NNOutput>(new NNOutput(*oldNNOutput));
  NNOutput* newNNOutput = newNNOutputSharedPtr->get();

  float* noisedPolicyProbs = new float[NNPos::MAX_NN_POLICY_SIZE];
  newNNOutput->noisedPolicyProbs = noisedPolicyProbs;

  for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++) 
    noisedPolicyProbs[i] = newNNOutput->getPolicyProb(i);

  if(searchParams.rootPolicyTemperature != 1.0 || searchParams.rootPolicyTemperatureEarly != 1.0) {
    double rootPolicyTemperature = interpolateEarly(
      searchParams.chosenMoveTemperatureHalflife, searchParams.rootPolicyTemperatureEarly, searchParams.rootPolicyTemperature
    );

    double maxValue = 0.0;
    for(int i = 0; i<policySize; i++) {
      double prob = noisedPolicyProbs[i];
      if(prob > maxValue)
        maxValue = prob;
    }
    assert(maxValue > 0.0);

    double logMaxValue = log(maxValue);
    double invTemp = 1.0 / rootPolicyTemperature;
    double sum = 0.0;

    for(int i = 0; i<policySize; i++) {
      if(noisedPolicyProbs[i] > 0) {
        //Numerically stable way to raise to power and normalize
        float p = (float)exp((log((double)noisedPolicyProbs[i]) - logMaxValue) * invTemp);
        noisedPolicyProbs[i] = p;
        sum += p;
      }
    }
    assert(sum > 0.0);
    for(int i = 0; i<policySize; i++) {
      if(noisedPolicyProbs[i] >= 0) {
        noisedPolicyProbs[i] = (float)(noisedPolicyProbs[i] / sum);
      }
    }
  }

  if(searchParams.rootNoiseEnabled) {
    addDirichletNoise(searchParams, thread.rand, policySize, noisedPolicyProbs);
  }

  if(avoidMoveUntilRescaleRoot) {
    const std::vector<int>& avoidMoveUntilByLoc =
      rootPla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
    if(avoidMoveUntilByLoc.size() > 0) {
      assert(avoidMoveUntilByLoc.size() >= Board::MAX_ARR_SIZE);
      double policySum = 0.0;
      for(Loc loc = 0; loc < Board::MAX_ARR_SIZE; loc++) {
        if((rootBoard.isOnBoard(loc) || loc == Board::PASS_LOC) && avoidMoveUntilByLoc[loc] <= 0) {
          int pos = getPos(loc);
          if(noisedPolicyProbs[pos] > 0) {
            policySum += noisedPolicyProbs[pos];
          }
        }
      }
      if(policySum > 0.0) {
        for(int i = 0; i < policySize; i++) {
          if(noisedPolicyProbs[i] > 0) {
            noisedPolicyProbs[i] = (float)(noisedPolicyProbs[i] / policySum);
          }
        }
      }
    }
  }
  //Move a small amount of policy to the hint move, around the same level that noising it would achieve
  if(rootHintLoc != Board::NULL_LOC) {
    const float propToMove = 0.02f;
    int pos = getPos(rootHintLoc);
    if(noisedPolicyProbs[pos] >= 0) {
      double amountToMove = 0.0;
      for(int i = 0; i<policySize; i++) {
        if(noisedPolicyProbs[i] >= 0) {
          amountToMove += noisedPolicyProbs[i] * propToMove;
          noisedPolicyProbs[i] *= (1.0f-propToMove);
        }
      }
      noisedPolicyProbs[pos] += (float)amountToMove;
    }
  }

  return newNNOutputSharedPtr;
}




double Search::getResultUtility(double winLossValue, double noResultValue) const {
  return (
    winLossValue * searchParams.winLossUtilityFactor +
    noResultValue * searchParams.noResultUtilityForWhite
  );
}

double Search::getResultUtilityFromNN(const NNOutput& nnOutput) const {
  return (
    (nnOutput.whiteWinProb - nnOutput.whiteLossProb) * searchParams.winLossUtilityFactor +
    nnOutput.whiteNoResultProb * searchParams.noResultUtilityForWhite
  );
}


double Search::getWhiteWinProbFromNN(const NNOutput& nnOutput, Player nextPla) const {
  assert(nextPla == P_WHITE || nextPla == P_BLACK);
  double c = searchParams.multiValueHeadUtilityMix;
  double multiHeadProb = nextPla == P_WHITE ?
    nnOutput.whiteWinProbByHead[3] :
    nnOutput.whiteWinProbByHead[2];
  return (1.0 - c) * nnOutput.whiteWinProbByHead[0] + c * multiHeadProb;
}


double Search::getBlackWinProbFromNN(const NNOutput& nnOutput, Player nextPla) const {
  assert(nextPla == P_WHITE || nextPla == P_BLACK);
  double c = searchParams.multiValueHeadUtilityMix;
  double multiHeadProb = nextPla == P_BLACK ?
    nnOutput.whiteLossProbByHead[3] :
    nnOutput.whiteLossProbByHead[2];
  return (1.0 - c) * nnOutput.whiteLossProbByHead[0] + c * multiHeadProb;
}


double Search::getWhiteWinUtility(double legacyUtility, double whiteWinProb) const {
  double c = searchParams.multiValueHeadUtilityMix;
  return (1.0 - c) * legacyUtility + c * (2.0 * whiteWinProb - 1.0);
}


double Search::getBlackWinUtilityInv(double legacyUtility, double blackWinProb) const {
  double c = searchParams.multiValueHeadUtilityMix;
  return (1.0 - c) * legacyUtility + c * (1.0 - 2.0 * blackWinProb);
}


double Search::getWhiteWinUtilityFromNN(const NNOutput& nnOutput, Player nextPla) const {
  double legacyUtility = getResultUtilityFromNN(nnOutput);
  return getWhiteWinUtility(legacyUtility, getWhiteWinProbFromNN(nnOutput,nextPla));
}


double Search::getBlackWinUtilityInvFromNN(const NNOutput& nnOutput, Player nextPla) const {
  double legacyUtility = getResultUtilityFromNN(nnOutput);
  return getBlackWinUtilityInv(legacyUtility, getBlackWinProbFromNN(nnOutput,nextPla));
}


double Search::getUtilityFromNN(const NNOutput& nnOutput, Player nextPla) const {
  double whiteWinUtility = getWhiteWinUtilityFromNN(nnOutput,nextPla);
  double blackWinUtilityInv = getBlackWinUtilityInvFromNN(nnOutput,nextPla);
  return 0.5 * (whiteWinUtility + blackWinUtilityInv);
}

void Search::getNormalObjectiveWorths(
  const SearchNode& node, double& winWorth, double& nonLossWorth
) const {
  double whiteWinUtility =
    node.stats.whiteWinUtilityAvg.load(std::memory_order_acquire);
  double blackWinUtilityInv =
    node.stats.blackWinUtilityInvAvg.load(std::memory_order_acquire);
  double sideWinUtility = node.nextPla == P_WHITE ?
    whiteWinUtility : -blackWinUtilityInv;
  double sideNonLossUtility = node.nextPla == P_WHITE ?
    blackWinUtilityInv : -whiteWinUtility;
  double winProb = std::clamp(0.5 * (sideWinUtility + 1.0),0.0,1.0);
  double nonLossProb =
    std::clamp(0.5 * (sideNonLossUtility + 1.0),0.0,1.0);

  //The common unresolved factor makes already-won positions less urgent.
  //The floor keeps either objective alive even when the current estimates
  //look resolved, since hidden tactics are exactly where head0 is least useful.
  constexpr double OBJECTIVE_WORTH_FLOOR = 0.10;
  double unresolvedWin = 1.0 - winProb;
  winWorth =
    OBJECTIVE_WORTH_FLOOR + nonLossProb * unresolvedWin;
  nonLossWorth =
    OBJECTIVE_WORTH_FLOOR + (1.0 - nonLossProb) * unresolvedWin;
}

double Search::getSideToMoveWinObjectiveWeight(const SearchNode& node) const {
  double p = searchParams.multiValueHeadSelectionBias;
  double fixedWinWeight = 0.5 + 0.5 * p;
  if(searchParams.multiHeadObjectiveSearchStrength <= 0.0)
    return fixedWinWeight;

  double winWorth;
  double nonLossWorth;
  getNormalObjectiveWorths(node,winWorth,nonLossWorth);
  double selectionPower = searchParams.multiHeadObjectiveSelectionPower;
  double poweredWinWorth = pow(winWorth,selectionPower);
  double poweredNonLossWorth = pow(nonLossWorth,selectionPower);
  double biasedWinWorth = fixedWinWeight * poweredWinWorth;
  double biasedNonLossWorth =
    (1.0 - fixedWinWeight) * poweredNonLossWorth;
  double dynamicWinWeight =
    biasedWinWorth + biasedNonLossWorth > 0.0 ?
      biasedWinWorth / (biasedWinWorth + biasedNonLossWorth) :
      fixedWinWeight;
  double strength = searchParams.multiHeadObjectiveSearchStrength;
  return
    (1.0 - strength) * fixedWinWeight +
    strength * dynamicWinWeight;
}

bool Search::isAllowedRootMove(Loc moveLoc) const {
  assert(moveLoc == Board::PASS_LOC || rootBoard.isOnBoard(moveLoc));

  if(searchParams.rootPruneUselessMoves &&
     rootHistory.moveHistory.size() > 0 
  ) {
    //nothing to write until now
  }

  if(searchParams.rootSymmetryPruning && moveLoc != Board::PASS_LOC && rootSymDupLoc[moveLoc]) {
    return false;
  }

  return true;
}

double Search::getPatternBonus(Hash128 patternBonusHash, Player prevMovePla) const {
  if(patternBonusTable == NULL || prevMovePla != plaThatSearchIsFor)
    return 0;
  return patternBonusTable->get(patternBonusHash).utilityBonus;
}

double Search::interpolateEarly(double halflife, double earlyValue, double value) const {
  double rawHalflives = (rootHistory.initialTurnNumber + rootHistory.moveHistory.size()) / halflife;
  double halflives = rawHalflives * 19.0 / sqrt(rootBoard.x_size*rootBoard.y_size);
  return value + (earlyValue - value) * pow(0.5, halflives);
}

void Search::getSelfUtilityLCBAndRadius(const SearchNode& parent, const SearchNode* child, int64_t edgeVisits, Loc moveLoc, double& lcbBuf, double& radiusBuf) const {
  (void)moveLoc;
  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  double utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);
  double utilitySqAvg = child->stats.utilitySqAvg.load(std::memory_order_acquire);
  double weightSum = child->stats.getChildWeight(edgeVisits,childVisits);
  double weightSqSum = child->stats.getChildWeightSq(edgeVisits,childVisits);

  // Max radius of the entire utility range
  double utilityRangeRadius = searchParams.winLossUtilityFactor;
  radiusBuf = 2.0 * utilityRangeRadius * searchParams.lcbStdevs;
  lcbBuf = -radiusBuf;
  if(childVisits <= 0 || weightSum <= 0.0 || weightSqSum <= 0.0)
    return;

  // Effective sample size for weighted data
  double ess = weightSum * weightSum / weightSqSum;

  // To behave well at low playouts, we'd like a variance approximation that makes sense even with very small sample sizes.
  // We'd like to avoid using a T distribution approximation because we actually know a bound on the scale of the utilities
  // involved, namely utilityRangeRadius. So instead add a prior with a small weight that the variance is the largest it can be.
  // This should give a relatively smooth scaling that works for small discrete samples but diminishes for larger playouts.
  double priorWeight = weightSum / (ess * ess * ess);
  utilitySqAvg = std::max(utilitySqAvg, utilityAvg * utilityAvg + 1e-8);
  utilitySqAvg = (utilitySqAvg * weightSum + (utilitySqAvg + utilityRangeRadius * utilityRangeRadius) * priorWeight) / (weightSum + priorWeight);
  weightSum += priorWeight;
  weightSqSum += priorWeight*priorWeight;

  // Recompute effective sample size now that we have the prior
  ess = weightSum * weightSum / weightSqSum;

  double utilityWithBonus = utilityAvg;
  double selfUtility = parent.nextPla == P_WHITE ? utilityWithBonus : -utilityWithBonus;

  double utilityVariance = utilitySqAvg - utilityAvg * utilityAvg;
  double estimateStdev = sqrt(utilityVariance / ess);
  double radius = estimateStdev * searchParams.lcbStdevs;

  lcbBuf = selfUtility - radius;
  radiusBuf = radius;
}
