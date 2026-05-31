#include "../search/search.h"

#include "../search/searchnode.h"
#include "../search/distributiontable.h"

//------------------------
#include "../core/using.h"
//------------------------



void Search::addLeafValue(
  SearchNode& node,
  double whiteWinProb,
  double blackWinProb,
  double noResultValue,
  double legacyUtility,
  double whiteWinUtility,
  double blackWinUtilityInv,
  double weight,
  double whiteWinWeight,
  double blackWinWeight,
  bool isTerminal,
  bool assumeNoExistingWeight
) {
  double winLossValue = whiteWinProb - blackWinProb;

  if(searchParams.subtreeValueBiasFactor != 0 && !isTerminal && node.subtreeValueBiasTableEntry != nullptr) {
    SubtreeValueBiasEntry& entry = *(node.subtreeValueBiasTableEntry);
    while(entry.entryLock.test_and_set(std::memory_order_acquire));
    double newEntryDeltaUtilitySum = entry.deltaUtilitySum;
    double newEntryWeightSum = entry.weightSum;
    entry.entryLock.clear(std::memory_order_release);
    //This is the amount of the direct evaluation of this node that we are going to bias towards the table entry
    const double biasFactor = searchParams.subtreeValueBiasFactor;
    if(newEntryWeightSum > 0.001) {
      double utilityBias = biasFactor * newEntryDeltaUtilitySum / newEntryWeightSum;
      legacyUtility += utilityBias;
      whiteWinUtility += utilityBias;
      blackWinUtilityInv += utilityBias;
    }
  }

  double patternBonus = getPatternBonus(node.patternBonusHash,getOpp(node.nextPla));
  legacyUtility += patternBonus;
  whiteWinUtility += patternBonus;
  blackWinUtilityInv += patternBonus;

  double utility = 0.5 * (whiteWinUtility + blackWinUtilityInv);
  double utilitySq = utility * utility;
  double weightSq = weight * weight;
  double whiteWinUtilitySq = whiteWinUtility * whiteWinUtility;
  double blackWinUtilityInvSq = blackWinUtilityInv * blackWinUtilityInv;
  double whiteWinWeightSq = whiteWinWeight * whiteWinWeight;
  double blackWinWeightSq = blackWinWeight * blackWinWeight;

  if(assumeNoExistingWeight) {
    while(node.statsLock.test_and_set(std::memory_order_acquire));
    node.stats.winLossValueAvg.store(winLossValue,std::memory_order_release);
    node.stats.noResultValueAvg.store(noResultValue,std::memory_order_release);
    node.stats.whiteWinProbAvg.store(whiteWinProb,std::memory_order_release);
    node.stats.blackWinProbAvg.store(blackWinProb,std::memory_order_release);
    node.stats.utilityAvg.store(utility,std::memory_order_release);
    node.stats.utilitySqAvg.store(utilitySq,std::memory_order_release);
    node.stats.weightSqSum.store(weightSq,std::memory_order_release);
    node.stats.weightSum.store(weight,std::memory_order_release);
    node.stats.whiteWinUtilityAvg.store(whiteWinUtility,std::memory_order_release);
    node.stats.whiteWinUtilitySqAvg.store(whiteWinUtilitySq,std::memory_order_release);
    node.stats.whiteWinWeightSqSum.store(whiteWinWeightSq,std::memory_order_release);
    node.stats.whiteWinWeightSum.store(whiteWinWeight,std::memory_order_release);
    node.stats.blackWinUtilityInvAvg.store(blackWinUtilityInv,std::memory_order_release);
    node.stats.blackWinUtilityInvSqAvg.store(blackWinUtilityInvSq,std::memory_order_release);
    node.stats.blackWinWeightSqSum.store(blackWinWeightSq,std::memory_order_release);
    node.stats.blackWinWeightSum.store(blackWinWeight,std::memory_order_release);
    int64_t oldVisits = node.stats.visits.fetch_add(1,std::memory_order_release);
    node.statsLock.clear(std::memory_order_release);
    // This should only be possible in the extremely rare case that we transpose to a terminal node from a non-terminal node probably due to
    // a hash collision, or that we have a graph history interaction that somehow changes whether a particular path ends the game or not, despite
    // our simpleRepetitionBoundGt logic... such that the node managed to get visits as a terminal node despite not having an nn eval. There's
    // nothing reasonable to do here once we have such a bad collision, so just at least don't crash.
    if(oldVisits != 0) {
      logger->write("WARNING: assumeNoExistingWeight for leaf but leaf already has visits");
    }
  }
  else {
    while(node.statsLock.test_and_set(std::memory_order_acquire));
    double oldWeightSum = node.stats.weightSum.load(std::memory_order_relaxed);
    double newWeightSum = oldWeightSum + weight;
    double oldWhiteWinWeightSum = node.stats.whiteWinWeightSum.load(std::memory_order_relaxed);
    double newWhiteWinWeightSum = oldWhiteWinWeightSum + whiteWinWeight;
    double oldBlackWinWeightSum = node.stats.blackWinWeightSum.load(std::memory_order_relaxed);
    double newBlackWinWeightSum = oldBlackWinWeightSum + blackWinWeight;

    node.stats.whiteWinProbAvg.store((node.stats.whiteWinProbAvg.load(std::memory_order_relaxed) * oldWhiteWinWeightSum + whiteWinProb * whiteWinWeight)/newWhiteWinWeightSum,std::memory_order_release);
    node.stats.blackWinProbAvg.store((node.stats.blackWinProbAvg.load(std::memory_order_relaxed) * oldBlackWinWeightSum + blackWinProb * blackWinWeight)/newBlackWinWeightSum,std::memory_order_release);
    double newWinLossValueAvg = node.stats.whiteWinProbAvg.load(std::memory_order_relaxed) - node.stats.blackWinProbAvg.load(std::memory_order_relaxed);
    node.stats.winLossValueAvg.store(newWinLossValueAvg,std::memory_order_release);
    node.stats.noResultValueAvg.store((node.stats.noResultValueAvg.load(std::memory_order_relaxed) * oldWeightSum + noResultValue * weight)/newWeightSum,std::memory_order_release);
    node.stats.utilityAvg.store((node.stats.utilityAvg.load(std::memory_order_relaxed) * oldWeightSum + utility * weight)/newWeightSum,std::memory_order_release);
    node.stats.utilitySqAvg.store((node.stats.utilitySqAvg.load(std::memory_order_relaxed) * oldWeightSum + utilitySq * weight)/newWeightSum,std::memory_order_release);
    node.stats.weightSqSum.store(node.stats.weightSqSum.load(std::memory_order_relaxed) + weightSq,std::memory_order_release);
    node.stats.weightSum.store(newWeightSum,std::memory_order_release);
    node.stats.whiteWinUtilityAvg.store((node.stats.whiteWinUtilityAvg.load(std::memory_order_relaxed) * oldWhiteWinWeightSum + whiteWinUtility * whiteWinWeight)/newWhiteWinWeightSum,std::memory_order_release);
    node.stats.whiteWinUtilitySqAvg.store((node.stats.whiteWinUtilitySqAvg.load(std::memory_order_relaxed) * oldWhiteWinWeightSum + whiteWinUtilitySq * whiteWinWeight)/newWhiteWinWeightSum,std::memory_order_release);
    node.stats.whiteWinWeightSqSum.store(node.stats.whiteWinWeightSqSum.load(std::memory_order_relaxed) + whiteWinWeightSq,std::memory_order_release);
    node.stats.whiteWinWeightSum.store(newWhiteWinWeightSum,std::memory_order_release);
    node.stats.blackWinUtilityInvAvg.store((node.stats.blackWinUtilityInvAvg.load(std::memory_order_relaxed) * oldBlackWinWeightSum + blackWinUtilityInv * blackWinWeight)/newBlackWinWeightSum,std::memory_order_release);
    node.stats.blackWinUtilityInvSqAvg.store((node.stats.blackWinUtilityInvSqAvg.load(std::memory_order_relaxed) * oldBlackWinWeightSum + blackWinUtilityInvSq * blackWinWeight)/newBlackWinWeightSum,std::memory_order_release);
    node.stats.blackWinWeightSqSum.store(node.stats.blackWinWeightSqSum.load(std::memory_order_relaxed) + blackWinWeightSq,std::memory_order_release);
    node.stats.blackWinWeightSum.store(newBlackWinWeightSum,std::memory_order_release);
    node.stats.visits.fetch_add(1,std::memory_order_release);
    node.statsLock.clear(std::memory_order_release);
  }
}

void Search::addCurrentNNOutputAsLeafValue(SearchNode& node, bool assumeNoExistingWeight) {
  const NNOutput* nnOutput = node.getNNOutput();
  assert(nnOutput != NULL);
  //Values in the search are from the perspective of white positive always
  double whiteWinProb = getWhiteWinProbFromNN(*nnOutput);
  double blackWinProb = getBlackWinProbFromNN(*nnOutput);
  double noResultProb = (double)nnOutput->whiteNoResultProb;
  double legacyUtility = getResultUtilityFromNN(*nnOutput);
  double whiteWinUtility = getWhiteWinUtility(legacyUtility, whiteWinProb);
  double blackWinUtilityInv = getBlackWinUtilityInv(legacyUtility, blackWinProb);
  double weight = computeWeightFromNNOutput(nnOutput);
  double whiteWinWeight = computeWhiteWinWeightFromNNOutput(nnOutput);
  double blackWinWeight = computeBlackWinWeightFromNNOutput(nnOutput);
  addLeafValue(node,whiteWinProb,blackWinProb,noResultProb,legacyUtility,whiteWinUtility,blackWinUtilityInv,weight,whiteWinWeight,blackWinWeight,false,assumeNoExistingWeight);
}

double Search::computeWeightFromNNOutput(const NNOutput* nnOutput) const {
  if(!searchParams.useUncertainty)
    return 1.0;
  if(!nnEvaluator->supportsShorttermError())
    return 1.0;

  double utilityUncertaintyWL = searchParams.winLossUtilityFactor * nnOutput->shorttermWinlossError;
  double utilityUncertainty = utilityUncertaintyWL;

  double poweredUncertainty;
  if(searchParams.uncertaintyExponent == 1.0)
    poweredUncertainty = utilityUncertainty;
  else if(searchParams.uncertaintyExponent == 0.5)
    poweredUncertainty = sqrt(utilityUncertainty);
  else
    poweredUncertainty = pow(utilityUncertainty, searchParams.uncertaintyExponent);

  double baselineUncertainty = searchParams.uncertaintyCoeff / searchParams.uncertaintyMaxWeight;
  double weight = searchParams.uncertaintyCoeff / (poweredUncertainty + baselineUncertainty);
  return weight;
}


double Search::computeWhiteWinWeightFromNNOutput(const NNOutput* nnOutput) const {
  return computeWeightFromNNOutput(nnOutput);
}


double Search::computeBlackWinWeightFromNNOutput(const NNOutput* nnOutput) const {
  return computeWeightFromNNOutput(nnOutput);
}

void Search::updateStatsAfterPlayout(SearchNode& node, SearchThread& thread, bool isRoot) {
  //The thread that grabs a 0 from this peforms the recomputation of stats.
  int32_t oldDirtyCounter = node.dirtyCounter.fetch_add(1,std::memory_order_acq_rel);
  assert(oldDirtyCounter >= 0);
  //If we atomically grab a nonzero, then we know another thread must already be doing the work, so we can skip the update ourselves.
  if(oldDirtyCounter > 0)
    return;
  int32_t numVisitsCompleted = 1;
  while(true) {
    //Perform update
    recomputeNodeStats(node,thread,numVisitsCompleted,isRoot);
    //Now attempt to undo the counter
    oldDirtyCounter = node.dirtyCounter.fetch_add(-numVisitsCompleted,std::memory_order_acq_rel);
    int32_t newDirtyCounter = oldDirtyCounter - numVisitsCompleted;
    //If no other threads incremented it in the meantime, so our decrement hits zero, we're done.
    if(newDirtyCounter <= 0) {
      assert(newDirtyCounter == 0);
      break;
    }
    //Otherwise, more threads incremented this more in the meantime. So we need to loop again and add their visits, recomputing again.
    numVisitsCompleted = newDirtyCounter;
    continue;
  }
}

//Recompute all the stats of this node based on its children, except its visits and virtual losses, which are not child-dependent and
//are updated in the manner specified.
//Assumes this node has an nnOutput
void Search::recomputeNodeStats(SearchNode& node, SearchThread& thread, int numVisitsToAdd, bool isRoot) {
  //Find all children and compute weighting of the children based on their values
  vector<MoreNodeStats>& statsBuf = thread.statsBuf;
  int numGoodChildren = 0;

  int childrenCapacity;
  const SearchChildPointer* children = node.getChildren(childrenCapacity);
  double origTotalChildWeight = 0.0;
  double origTotalWhiteWinChildWeight = 0.0;
  double origTotalBlackWinChildWeight = 0.0;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    MoreNodeStats& stats = statsBuf[numGoodChildren];

    Loc moveLoc = children[i].getMoveLocRelaxed();
    int64_t edgeVisits = children[i].getEdgeVisits();
    stats.stats = NodeStats(child->stats);

    if(stats.stats.visits <= 0 || stats.stats.weightSum <= 0.0 || edgeVisits <= 0)
      continue;

    double childUtility = stats.stats.utilityAvg;
    stats.selfUtility = node.nextPla == P_WHITE ? childUtility : -childUtility;
    stats.weightAdjusted = stats.stats.getChildWeight(edgeVisits);
    stats.whiteWinSelfUtility = node.nextPla == P_WHITE ? stats.stats.whiteWinUtilityAvg : -stats.stats.whiteWinUtilityAvg;
    stats.whiteWinWeightAdjusted = stats.stats.getChildWhiteWinWeight(edgeVisits);
    stats.blackWinSelfUtility = node.nextPla == P_WHITE ? stats.stats.blackWinUtilityInvAvg : -stats.stats.blackWinUtilityInvAvg;
    stats.blackWinWeightAdjusted = stats.stats.getChildBlackWinWeight(edgeVisits);
    stats.prevMoveLoc = moveLoc;

    if(stats.whiteWinWeightAdjusted <= 0.0)
      stats.whiteWinWeightAdjusted = stats.weightAdjusted;
    if(stats.blackWinWeightAdjusted <= 0.0)
      stats.blackWinWeightAdjusted = stats.weightAdjusted;

    origTotalChildWeight += stats.weightAdjusted;
    origTotalWhiteWinChildWeight += stats.whiteWinWeightAdjusted;
    origTotalBlackWinChildWeight += stats.blackWinWeightAdjusted;
    numGoodChildren++;
  }

  const NNOutput* nodeNNOutput = node.getNNOutput();
  assert(nodeNNOutput != NULL);

  auto adjustObjectiveWeights = [&](int objective, double origTotalWeight) {
    double currentTotalWeight = origTotalWeight;
    for(int i = 0; i<numGoodChildren; i++) {
      if(objective == 0) {
        statsBuf[i].selfUtility = node.nextPla == P_WHITE ? statsBuf[i].stats.utilityAvg : -statsBuf[i].stats.utilityAvg;
      }
      else if(objective == 1) {
        statsBuf[i].selfUtility = statsBuf[i].whiteWinSelfUtility;
        statsBuf[i].weightAdjusted = statsBuf[i].whiteWinWeightAdjusted;
      }
      else {
        statsBuf[i].selfUtility = statsBuf[i].blackWinSelfUtility;
        statsBuf[i].weightAdjusted = statsBuf[i].blackWinWeightAdjusted;
      }
    }

    if(searchParams.useNoisePruning && numGoodChildren > 0) {
      double policyProbsBuf[NNPos::MAX_NN_POLICY_SIZE];
      for(int i = 0; i<numGoodChildren; i++)
        policyProbsBuf[i] =
          std::max(1e-30, (double)nodeNNOutput->getPolicyProbMaybeNoised(getPos(statsBuf[i].prevMoveLoc)));
      currentTotalWeight = pruneNoiseWeight(statsBuf, numGoodChildren, currentTotalWeight, policyProbsBuf);
    }

    {
      double amountToSubtract = 0.0;
      double amountToPrune = 0.0;
      if(isRoot && searchParams.rootNoiseEnabled && !searchParams.useNoisePruning) {
        double maxChildWeight = 0.0;
        for(int i = 0; i<numGoodChildren; i++) {
          if(statsBuf[i].weightAdjusted > maxChildWeight)
            maxChildWeight = statsBuf[i].weightAdjusted;
        }
        amountToSubtract = std::min(searchParams.chosenMoveSubtract, maxChildWeight/64.0);
        amountToPrune = std::min(searchParams.chosenMovePrune, maxChildWeight/64.0);
      }

      downweightBadChildrenAndNormalizeWeight(
        numGoodChildren, currentTotalWeight, currentTotalWeight,
        amountToSubtract, amountToPrune, statsBuf
      );
    }

    for(int i = 0; i<numGoodChildren; i++) {
      if(objective == 1)
        statsBuf[i].whiteWinWeightAdjusted = statsBuf[i].weightAdjusted;
      else if(objective == 2)
        statsBuf[i].blackWinWeightAdjusted = statsBuf[i].weightAdjusted;
    }
    return currentTotalWeight;
  };

  double currentTotalChildWeight = adjustObjectiveWeights(0, origTotalChildWeight);
  double legacyWeightAdjustedBuf[NNPos::MAX_NN_POLICY_SIZE];
  for(int i = 0; i<numGoodChildren; i++)
    legacyWeightAdjustedBuf[i] = statsBuf[i].weightAdjusted;
  double currentTotalWhiteWinChildWeight;
  double currentTotalBlackWinChildWeight;
  if(searchParams.multiValueHeadUtilityMix == 0.0) {
    currentTotalWhiteWinChildWeight = currentTotalChildWeight;
    currentTotalBlackWinChildWeight = currentTotalChildWeight;
    for(int i = 0; i<numGoodChildren; i++) {
      statsBuf[i].whiteWinWeightAdjusted = legacyWeightAdjustedBuf[i];
      statsBuf[i].blackWinWeightAdjusted = legacyWeightAdjustedBuf[i];
    }
  }
  else {
    currentTotalWhiteWinChildWeight = adjustObjectiveWeights(1, origTotalWhiteWinChildWeight);
    currentTotalBlackWinChildWeight = adjustObjectiveWeights(2, origTotalBlackWinChildWeight);
  }

  double whiteWinProbSum = 0.0;
  double blackWinProbSum = 0.0;
  double noResultValueSum = 0.0;
  double utilitySum = 0.0;
  double utilitySqSum = 0.0;
  double weightSqSum = 0.0;
  double weightSum = currentTotalChildWeight;
  double whiteWinUtilitySum = 0.0;
  double whiteWinUtilitySqSum = 0.0;
  double whiteWinWeightSqSum = 0.0;
  double whiteWinWeightSum = currentTotalWhiteWinChildWeight;
  double blackWinUtilityInvSum = 0.0;
  double blackWinUtilityInvSqSum = 0.0;
  double blackWinWeightSqSum = 0.0;
  double blackWinWeightSum = currentTotalBlackWinChildWeight;

  for(int i = 0; i<numGoodChildren; i++) {
    const NodeStats& stats = statsBuf[i].stats;

    double desiredWeight = legacyWeightAdjustedBuf[i];
    double weightScaling = desiredWeight / stats.weightSum;
    noResultValueSum += desiredWeight * stats.noResultValueAvg;
    utilitySum += desiredWeight * stats.utilityAvg;
    utilitySqSum += desiredWeight * stats.utilitySqAvg;
    weightSqSum += weightScaling * weightScaling * stats.weightSqSum;

    double desiredWhiteWinWeight = statsBuf[i].whiteWinWeightAdjusted;
    double whiteWinWeightScaling = desiredWhiteWinWeight / stats.whiteWinWeightSum;
    whiteWinProbSum += desiredWhiteWinWeight * stats.whiteWinProbAvg;
    whiteWinUtilitySum += desiredWhiteWinWeight * stats.whiteWinUtilityAvg;
    whiteWinUtilitySqSum += desiredWhiteWinWeight * stats.whiteWinUtilitySqAvg;
    whiteWinWeightSqSum += whiteWinWeightScaling * whiteWinWeightScaling * stats.whiteWinWeightSqSum;

    double desiredBlackWinWeight = statsBuf[i].blackWinWeightAdjusted;
    double blackWinWeightScaling = desiredBlackWinWeight / stats.blackWinWeightSum;
    blackWinProbSum += desiredBlackWinWeight * stats.blackWinProbAvg;
    blackWinUtilityInvSum += desiredBlackWinWeight * stats.blackWinUtilityInvAvg;
    blackWinUtilityInvSqSum += desiredBlackWinWeight * stats.blackWinUtilityInvSqAvg;
    blackWinWeightSqSum += blackWinWeightScaling * blackWinWeightScaling * stats.blackWinWeightSqSum;
  }

  //Also add in the direct evaluation of this node.
  {
    double whiteWinProb = getWhiteWinProbFromNN(*nodeNNOutput);
    double blackWinProb = getBlackWinProbFromNN(*nodeNNOutput);
    double noResultProb = (double)nodeNNOutput->whiteNoResultProb;
    double legacyUtility = getResultUtilityFromNN(*nodeNNOutput);
    double whiteWinUtility = getWhiteWinUtility(legacyUtility, whiteWinProb);
    double blackWinUtilityInv = getBlackWinUtilityInv(legacyUtility, blackWinProb);
    double utility = 0.5 * (whiteWinUtility + blackWinUtilityInv);

    if(searchParams.subtreeValueBiasFactor != 0 && node.subtreeValueBiasTableEntry != nullptr) {
      SubtreeValueBiasEntry& entry = *(node.subtreeValueBiasTableEntry);

      double newEntryDeltaUtilitySum;
      double newEntryWeightSum;

      if(currentTotalChildWeight > 1e-10) {
        double utilityChildren = utilitySum / currentTotalChildWeight;
        double subtreeValueBiasWeight = pow(origTotalChildWeight, searchParams.subtreeValueBiasWeightExponent);
        double subtreeValueBiasDeltaSum = (utilityChildren - utility) * subtreeValueBiasWeight;

        while(entry.entryLock.test_and_set(std::memory_order_acquire));
        entry.deltaUtilitySum += subtreeValueBiasDeltaSum - node.lastSubtreeValueBiasDeltaSum;
        entry.weightSum += subtreeValueBiasWeight - node.lastSubtreeValueBiasWeight;
        newEntryDeltaUtilitySum = entry.deltaUtilitySum;
        newEntryWeightSum = entry.weightSum;
        node.lastSubtreeValueBiasDeltaSum = subtreeValueBiasDeltaSum;
        node.lastSubtreeValueBiasWeight = subtreeValueBiasWeight;
        entry.entryLock.clear(std::memory_order_release);
      }
      else {
        while(entry.entryLock.test_and_set(std::memory_order_acquire));
        newEntryDeltaUtilitySum = entry.deltaUtilitySum;
        newEntryWeightSum = entry.weightSum;
        entry.entryLock.clear(std::memory_order_release);
      }

      //This is the amount of the direct evaluation of this node that we are going to bias towards the table entry
      const double biasFactor = searchParams.subtreeValueBiasFactor;
      if(newEntryWeightSum > 0.001) {
        double utilityBias = biasFactor * newEntryDeltaUtilitySum / newEntryWeightSum;
        legacyUtility += utilityBias;
        whiteWinUtility += utilityBias;
        blackWinUtilityInv += utilityBias;
        utility += utilityBias;
      }
    }

    double weight = computeWeightFromNNOutput(nodeNNOutput);
    double whiteWinWeight = computeWhiteWinWeightFromNNOutput(nodeNNOutput);
    double blackWinWeight = computeBlackWinWeightFromNNOutput(nodeNNOutput);
    whiteWinProbSum += whiteWinProb * whiteWinWeight;
    blackWinProbSum += blackWinProb * blackWinWeight;
    noResultValueSum += noResultProb * weight;
    utilitySum += utility * weight;
    utilitySqSum += utility * utility * weight;
    weightSqSum += weight * weight;
    weightSum += weight;
    whiteWinUtilitySum += whiteWinUtility * whiteWinWeight;
    whiteWinUtilitySqSum += whiteWinUtility * whiteWinUtility * whiteWinWeight;
    whiteWinWeightSqSum += whiteWinWeight * whiteWinWeight;
    whiteWinWeightSum += whiteWinWeight;
    blackWinUtilityInvSum += blackWinUtilityInv * blackWinWeight;
    blackWinUtilityInvSqSum += blackWinUtilityInv * blackWinUtilityInv * blackWinWeight;
    blackWinWeightSqSum += blackWinWeight * blackWinWeight;
    blackWinWeightSum += blackWinWeight;
  }

  double whiteWinProbAvg = whiteWinProbSum / whiteWinWeightSum;
  double blackWinProbAvg = blackWinProbSum / blackWinWeightSum;
  double winLossValueAvg = whiteWinProbAvg - blackWinProbAvg;
  double noResultValueAvg = noResultValueSum / weightSum;
  double utilityAvg = utilitySum / weightSum;
  double utilitySqAvg = utilitySqSum / weightSum;
  double whiteWinUtilityAvg = whiteWinUtilitySum / whiteWinWeightSum;
  double whiteWinUtilitySqAvg = whiteWinUtilitySqSum / whiteWinWeightSum;
  double blackWinUtilityInvAvg = blackWinUtilityInvSum / blackWinWeightSum;
  double blackWinUtilityInvSqAvg = blackWinUtilityInvSqSum / blackWinWeightSum;

  double patternBonus = getPatternBonus(node.patternBonusHash,getOpp(node.nextPla));
  double oldUtilityAvg = utilityAvg;
  utilityAvg += patternBonus;
  utilitySqAvg = utilitySqAvg + (utilityAvg * utilityAvg - oldUtilityAvg * oldUtilityAvg);

  double oldWhiteWinUtilityAvg = whiteWinUtilityAvg;
  whiteWinUtilityAvg += patternBonus;
  whiteWinUtilitySqAvg = whiteWinUtilitySqAvg + (whiteWinUtilityAvg * whiteWinUtilityAvg - oldWhiteWinUtilityAvg * oldWhiteWinUtilityAvg);

  double oldBlackWinUtilityInvAvg = blackWinUtilityInvAvg;
  blackWinUtilityInvAvg += patternBonus;
  blackWinUtilityInvSqAvg = blackWinUtilityInvSqAvg + (blackWinUtilityInvAvg * blackWinUtilityInvAvg - oldBlackWinUtilityInvAvg * oldBlackWinUtilityInvAvg);

  //TODO statslock may be unnecessary now with the dirtyCounter mechanism?
  while(node.statsLock.test_and_set(std::memory_order_acquire));
  node.stats.winLossValueAvg.store(winLossValueAvg,std::memory_order_release);
  node.stats.noResultValueAvg.store(noResultValueAvg,std::memory_order_release);
  node.stats.whiteWinProbAvg.store(whiteWinProbAvg,std::memory_order_release);
  node.stats.blackWinProbAvg.store(blackWinProbAvg,std::memory_order_release);
  node.stats.utilityAvg.store(utilityAvg,std::memory_order_release);
  node.stats.utilitySqAvg.store(utilitySqAvg,std::memory_order_release);
  node.stats.weightSqSum.store(weightSqSum,std::memory_order_release);
  node.stats.weightSum.store(weightSum,std::memory_order_release);
  node.stats.whiteWinUtilityAvg.store(whiteWinUtilityAvg,std::memory_order_release);
  node.stats.whiteWinUtilitySqAvg.store(whiteWinUtilitySqAvg,std::memory_order_release);
  node.stats.whiteWinWeightSqSum.store(whiteWinWeightSqSum,std::memory_order_release);
  node.stats.whiteWinWeightSum.store(whiteWinWeightSum,std::memory_order_release);
  node.stats.blackWinUtilityInvAvg.store(blackWinUtilityInvAvg,std::memory_order_release);
  node.stats.blackWinUtilityInvSqAvg.store(blackWinUtilityInvSqAvg,std::memory_order_release);
  node.stats.blackWinWeightSqSum.store(blackWinWeightSqSum,std::memory_order_release);
  node.stats.blackWinWeightSum.store(blackWinWeightSum,std::memory_order_release);
  node.stats.visits.fetch_add(numVisitsToAdd,std::memory_order_release);
  node.statsLock.clear(std::memory_order_release);
}
void Search::downweightBadChildrenAndNormalizeWeight(
  int numChildren,
  double currentTotalWeight, //The current sum of statsBuf[i].weightAdjusted
  double desiredTotalWeight, //What statsBuf[i].weightAdjusted should sum up to after this function is done.
  double amountToSubtract,
  double amountToPrune,
  vector<MoreNodeStats>& statsBuf
) const {
  if(numChildren <= 0 || currentTotalWeight <= 0.0)
    return;

  if(searchParams.valueWeightExponent == 0 ) {
    for(int i = 0; i<numChildren; i++) {
      if(statsBuf[i].weightAdjusted < amountToPrune) {
        currentTotalWeight -= statsBuf[i].weightAdjusted;
        statsBuf[i].weightAdjusted = 0.0;
        continue;
      }
      double newWeight = statsBuf[i].weightAdjusted - amountToSubtract;
      if(newWeight <= 0) {
        currentTotalWeight -= statsBuf[i].weightAdjusted;
        statsBuf[i].weightAdjusted = 0.0;
      }
      else {
        currentTotalWeight -= amountToSubtract;
        statsBuf[i].weightAdjusted = newWeight;
      }
    }

    if(currentTotalWeight != desiredTotalWeight) {
      double factor = desiredTotalWeight / currentTotalWeight;
      for(int i = 0; i<numChildren; i++)
        statsBuf[i].weightAdjusted *= factor;
    }
    return;
  }

  assert(numChildren <= NNPos::MAX_NN_POLICY_SIZE);
  double stdevs[NNPos::MAX_NN_POLICY_SIZE];
  double simpleValueSum = 0.0;
  for(int i = 0; i<numChildren; i++) {
    int64_t numVisits = statsBuf[i].stats.visits;
    assert(numVisits >= 0);
    if(numVisits == 0)
      continue;

    double weight = statsBuf[i].weightAdjusted;
    double precision = 1.5 * sqrt(weight);

    //Ensure some minimum variance for stability regardless of how we change the above formula
    static const double minVariance = 0.00000001;
    stdevs[i] = sqrt(minVariance + 1.0 / precision);
    simpleValueSum += statsBuf[i].selfUtility * weight;
  }

  double simpleValue = simpleValueSum / currentTotalWeight;

  double totalNewUnnormWeight = 0.0;
  for(int i = 0; i<numChildren; i++) {
    if(statsBuf[i].stats.visits == 0)
      continue;

    if(statsBuf[i].weightAdjusted < amountToPrune) {
      currentTotalWeight -= statsBuf[i].weightAdjusted;
      statsBuf[i].weightAdjusted = 0.0;
      continue;
    }
    double newWeight = statsBuf[i].weightAdjusted - amountToSubtract;
    if(newWeight <= 0) {
      currentTotalWeight -= statsBuf[i].weightAdjusted;
      statsBuf[i].weightAdjusted = 0.0;
    }
    else {
      currentTotalWeight -= amountToSubtract;
      statsBuf[i].weightAdjusted = newWeight;
    }

    double z = (statsBuf[i].selfUtility - simpleValue) / stdevs[i];
    //Also just for numeric sanity, make sure everything has some tiny minimum value.
    double p = valueWeightDistribution->getCdf(z) + 0.0001;
    statsBuf[i].weightAdjusted *= pow(p, searchParams.valueWeightExponent);
    totalNewUnnormWeight += statsBuf[i].weightAdjusted;
  }

  //Post-process and normalize to sum to the desired weight
  assert(totalNewUnnormWeight > 0.0);
  double factor = desiredTotalWeight / totalNewUnnormWeight;
  for(int i = 0; i<numChildren; i++)
    statsBuf[i].weightAdjusted *= factor;
}


//Returns the new sum of weightAdjusted
double Search::pruneNoiseWeight(vector<MoreNodeStats>& statsBuf, int numChildren, double totalChildWeight, const double* policyProbsBuf) const {
  if(numChildren <= 1 || totalChildWeight <= 0.00001)
    return totalChildWeight;

  // Children are normally sorted in policy order in KataGo.
  // But this is not guaranteed, because at the root, we might recompute the nnoutput, or when finding the best new child, we have hacks like antiMirror policy
  // and other adjustments. For simplicity, we just consider children in sorted order anyways for this pruning, since it will be close.

  // For any child, if its own utility is lower than the weighted average utility of the children before it, it's downweighted if it exceeds much more than a
  // raw-policy share of the weight.
  double utilitySumSoFar = 0;
  double weightSumSoFar = 0;
  //double rawPolicyUtilitySumSoFar = 0;
  double rawPolicySumSoFar = 0;
  for(int i = 0; i<numChildren; i++) {
    double utility = statsBuf[i].selfUtility;
    double oldWeight = statsBuf[i].weightAdjusted;
    double rawPolicy = policyProbsBuf[i];

    double newWeight = oldWeight;
    if(weightSumSoFar > 0 && rawPolicySumSoFar > 0) {
      double avgUtilitySoFar = utilitySumSoFar / weightSumSoFar;
      double utilityGap = avgUtilitySoFar - utility;
      if(utilityGap > 0) {
        double weightShareFromRawPolicy = weightSumSoFar * rawPolicy / rawPolicySumSoFar;
        //If the child is more than double its proper share of the weight
        double lenientWeightShareFromRawPolicy = 2.0 * weightShareFromRawPolicy;
        if(oldWeight > lenientWeightShareFromRawPolicy) {
          double excessWeight = oldWeight - lenientWeightShareFromRawPolicy;
          double weightToSubtract = excessWeight * (1.0 - exp(-utilityGap / searchParams.noisePruneUtilityScale));
          if(weightToSubtract > searchParams.noisePruningCap)
            weightToSubtract = searchParams.noisePruningCap;

          newWeight = oldWeight - weightToSubtract;
          statsBuf[i].weightAdjusted = newWeight;
        }
      }
    }
    utilitySumSoFar += utility * newWeight;
    weightSumSoFar += newWeight;
    //rawPolicyUtilitySumSoFar += utility * rawPolicy;
    rawPolicySumSoFar += rawPolicy;
  }
  return weightSumSoFar;
}
