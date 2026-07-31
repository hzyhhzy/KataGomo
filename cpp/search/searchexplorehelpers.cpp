#include "../search/search.h"

#include "../search/searchnode.h"

//------------------------
#include "../core/using.h"
//------------------------

static double cpuctExploration(double totalChildWeight, const SearchParams& searchParams) {
  if(searchParams.cpuctExplorationLog >= 0)
    return searchParams.cpuctExploration +
           searchParams.cpuctExplorationLog *
             log((totalChildWeight + searchParams.cpuctExplorationBase) / searchParams.cpuctExplorationBase);
  else
    return searchParams.cpuctExploration *
           pow(
             (totalChildWeight + searchParams.cpuctExplorationBase) / searchParams.cpuctExplorationBase,
             -searchParams.cpuctExplorationLog);
}

inline double noResultUtilityDecrease(double x, double d, Color color)
{
  if (color == C_BLACK)
    d = -d;
  return x-tanh(atanh(x * 0.999999) - d);
}
//Tiny constant to add to numerator of puct formula to make it positive
//even when visits = 0.
static constexpr double TOTALCHILDWEIGHT_PUCT_OFFSET = 0.01;

double Search::getExploreScaling(
  double totalChildWeight, double parentUtilityStdevFactor
) const {
  return
    cpuctExploration(totalChildWeight, searchParams)
    * sqrt(totalChildWeight + TOTALCHILDWEIGHT_PUCT_OFFSET)
    * parentUtilityStdevFactor;
}

double Search::getExploreSelectionValue(
  double exploreScaling,
  double nnPolicyProb,
  double childWeight,
  double childUtility,
  Player pla
) const {
  if(nnPolicyProb < 0)
    return POLICY_ILLEGAL_SELECTION_VALUE;

  double exploreComponent = exploreScaling * nnPolicyProb / (1.0 + childWeight);

  //At the last moment, adjust value to be from the player's perspective, so that players prefer values in their favor
  //rather than in white's favor
  double valueComponent = pla == P_WHITE ? childUtility : -childUtility;
  return exploreComponent + valueComponent;
}

//Return the childWeight that would make Search::getExploreSelectionValue return the given explore selection value.
//Or return 0, if it would be less than 0.
double Search::getExploreSelectionValueInverse(
  double exploreSelectionValue,
  double exploreScaling,
  double nnPolicyProb,
  double childUtility,
  Player pla
) const {
  if(nnPolicyProb < 0)
    return 0;
  double valueComponent = pla == P_WHITE ? childUtility : -childUtility;

  double exploreComponent = exploreSelectionValue - valueComponent;
  double exploreComponentScaling = exploreScaling * nnPolicyProb;

  //Guard against float weirdness
  if(exploreComponent <= 0)
    return 1e100;

  double childWeight = exploreComponentScaling / exploreComponent - 1;
  if(childWeight < 0)
    childWeight = 0;
  return childWeight;
}

static void maybeApplyWideRootNoise(
  double& childUtility,
  float& nnPolicyProb,
  const SearchParams& searchParams,
  SearchThread* thread,
  const SearchNode& parent
) {
  //For very large wideRootNoise, go ahead and also smooth out the policy
  nnPolicyProb = (float)pow(nnPolicyProb, 1.0 / (4.0*searchParams.wideRootNoise + 1.0));
  if(thread->rand.nextBool(0.5)) {
    double bonus = searchParams.wideRootNoise * std::fabs(thread->rand.nextGaussian());
    if(parent.nextPla == P_WHITE)
      childUtility += bonus;
    else
      childUtility -= bonus;
  }
}


double Search::getExploreSelectionValueOfChild(
  const SearchNode& parent, float nnPolicyProb, const SearchNode* child,
  Loc moveLoc,
  double exploreScaling,
  double totalChildWeight, int64_t childEdgeVisits, double fpuValue,
  double parentUtility, double parentWeightPerVisit,
  bool isDuringSearch, double maxChildWeight, SearchThread* thread
) const {
  (void)parentUtility;

  int32_t childVirtualLosses = child->virtualLosses.load(std::memory_order_acquire);
  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  double utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);
  double noResultValueAvg = child->stats.noResultValueAvg.load(std::memory_order_acquire);
  double childWeight = child->stats.getChildWeight(childEdgeVisits,childVisits);

  //It's possible that childVisits is actually 0 here with multithreading because we're visiting this node while a child has
  //been expanded but its thread not yet finished its first visit.
  //It's also possible that we observe childWeight <= 0 even though childVisits >= due to multithreading, the two could
  //be out of sync briefly since they are separate atomics.
  double childUtility;
  if(childVisits <= 0 || childWeight <= 0.0)
    childUtility = fpuValue;
  else {
    double parentNoResultValueAvg = parent.stats.noResultValueAvg.load(std::memory_order_acquire);
    double d = searchParams.noResultUtilityReduce * (1 - parentNoResultValueAvg);
    childUtility =
      utilityAvg - noResultValueAvg * noResultUtilityDecrease(searchParams.noResultUtilityForWhite, d, parent.nextPla);

  }

  //Virtual losses to direct threads down different paths
  if(childVirtualLosses > 0) {
    double virtualLossWeight = childVirtualLosses * searchParams.numVirtualLossesPerThread;

    double utilityRadius = searchParams.winLossUtilityFactor;
    double virtualLossUtility = (parent.nextPla == P_WHITE ? -utilityRadius : utilityRadius);
    double virtualLossWeightFrac = (double)virtualLossWeight / (virtualLossWeight + std::max(0.25,childWeight));
    childUtility = childUtility + (virtualLossUtility - childUtility) * virtualLossWeightFrac;
    childWeight += virtualLossWeight;
  }

  if(isDuringSearch && (&parent == rootNode)) {
    //Futile visits pruning - skip this move if the amount of time we have left to search is too small, assuming
    //its average weight per visit is maintained.
    //We use childVisits rather than childEdgeVisits for the final estimate since when childEdgeVisits < childVisits, adding new visits is instant.
    if(searchParams.futileVisitsThreshold > 0) {
      double requiredWeight = searchParams.futileVisitsThreshold * maxChildWeight;
      //Avoid divide by 0 by adding a prior equal to the parent's weight per visit
      double averageVisitsPerWeight = (childEdgeVisits + 1.0) / (childWeight + parentWeightPerVisit);
      double estimatedRequiredVisits = requiredWeight * averageVisitsPerWeight;
      if(childVisits + thread->upperBoundVisitsLeft < estimatedRequiredVisits)
        return FUTILE_VISITS_PRUNE_VALUE;
    }
    //Hack to get the root to funnel more visits down child branches
    if(searchParams.rootDesiredPerChildVisitsCoeff > 0.0) {
      if(nnPolicyProb > 0 && childWeight < sqrt(nnPolicyProb * totalChildWeight * searchParams.rootDesiredPerChildVisitsCoeff)) {
        return 1e20;
      }
    }
    //Hack for hintloc - must search this move almost as often as the most searched move
    if(rootHintLoc != Board::NULL_LOC && moveLoc == rootHintLoc) {
      double averageWeightPerVisit = (childWeight + parentWeightPerVisit) / (childVisits + 1.0);
      int childrenCapacity;
      const SearchChildPointer* children = parent.getChildren(childrenCapacity);
      for(int i = 0; i<childrenCapacity; i++) {
        const SearchNode* c = children[i].getIfAllocated();
        if(c == NULL)
          break;
        int64_t cEdgeVisits = children[i].getEdgeVisits();
        double cWeight = c->stats.getChildWeight(cEdgeVisits);
        if(childWeight + averageWeightPerVisit < cWeight * 0.8)
          return 1e20;
      }
    }

    if(searchParams.wideRootNoise > 0.0 && nnPolicyProb >= 0) {
      maybeApplyWideRootNoise(childUtility, nnPolicyProb, searchParams, thread, parent);
    }
  }

  return getExploreSelectionValue(exploreScaling,nnPolicyProb,childWeight,childUtility,parent.nextPla);
}

double Search::getNewExploreSelectionValue(
  const SearchNode& parent,
  double exploreScaling,
  float nnPolicyProb,
  double fpuValue,
  double parentWeightPerVisit,
  double maxChildWeight, SearchThread* thread
) const {
  double childWeight = 0;
  double childUtility = fpuValue;
  if(&parent == rootNode) {
    //Futile visits pruning - skip this move if the amount of time we have left to search is too small
    if(searchParams.futileVisitsThreshold > 0) {
      //Avoid divide by 0 by adding a prior equal to the parent's weight per visit
      double averageVisitsPerWeight = 1.0 / parentWeightPerVisit;
      double requiredWeight = searchParams.futileVisitsThreshold * maxChildWeight;
      double estimatedRequiredVisits = requiredWeight * averageVisitsPerWeight;
      if(thread->upperBoundVisitsLeft < estimatedRequiredVisits)
        return FUTILE_VISITS_PRUNE_VALUE;
    }
    if(searchParams.wideRootNoise > 0.0) {
      maybeApplyWideRootNoise(childUtility, nnPolicyProb, searchParams, thread, parent);
    }
  }
  return getExploreSelectionValue(exploreScaling,nnPolicyProb,childWeight,childUtility,parent.nextPla);
}

double Search::getReducedPlaySelectionWeight(
  const SearchNode& parent, float nnPolicyProb, const SearchNode* child,
  Loc moveLoc,
  double exploreScaling,
  int64_t childEdgeVisits,
  double bestChildExploreSelectionValue
) const {
  assert(&parent == rootNode);

  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  double utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);
  double childWeight = child->stats.getChildWeight(childEdgeVisits,childVisits);

  //Child visits may be 0 if this function is called in a multithreaded context, such as during live analysis
  //Child weight may also be 0 if it's out of sync.
  if(childVisits <= 0 || childWeight <= 0.0)
    return 0;

  //Tiny adjustment for passing
  double childUtility = utilityAvg;
  double childWeightWeRetrospectivelyWanted = getExploreSelectionValueInverse(
    bestChildExploreSelectionValue, exploreScaling, nnPolicyProb, childUtility, parent.nextPla
  );
  if(childWeight > childWeightWeRetrospectivelyWanted)
    return childWeightWeRetrospectivelyWanted;
  return childWeight;
}

double Search::getFpuValueForChildrenAssumeVisitedByStats(
  const SearchNode& node, Player pla, bool isRoot, double policyProbMassVisited,
  double visits, double weightSum, double utilityAvg, double utilitySqAvg, double nnUtility,
  double& parentUtility, double& parentWeightPerVisit, double& parentUtilityStdevFactor
) const {
  assert(visits > 0);
  assert(weightSum > 0.0);
  parentWeightPerVisit = weightSum / visits;
  parentUtility = utilityAvg;
  double variancePrior = searchParams.cpuctUtilityStdevPrior * searchParams.cpuctUtilityStdevPrior;
  double variancePriorWeight = searchParams.cpuctUtilityStdevPriorWeight;
  double parentUtilityStdev;
  if(visits <= 0 || weightSum <= 1)
    parentUtilityStdev = searchParams.cpuctUtilityStdevPrior;
  else {
    double utilitySq = parentUtility * parentUtility;
    //Make sure we're robust to numerical precision issues or threading desync of these values, so we don't observe negative variance
    if(utilitySqAvg < utilitySq)
      utilitySqAvg = utilitySq;
    parentUtilityStdev = sqrt(
      std::max(
        0.0,
        ((utilitySq + variancePrior) * variancePriorWeight + utilitySqAvg * weightSum)
        / (variancePriorWeight + weightSum - 1.0)
        - utilitySq
      )
    );
  }
  parentUtilityStdevFactor = 1.0 + searchParams.cpuctUtilityStdevScale * (parentUtilityStdev / searchParams.cpuctUtilityStdevPrior - 1.0);

  double parentUtilityForFPU = parentUtility;
  if(searchParams.fpuParentWeightByVisitedPolicy) {
    double avgWeight = std::min(1.0, pow(policyProbMassVisited, searchParams.fpuParentWeightByVisitedPolicyPow));
    parentUtilityForFPU = avgWeight * parentUtility + (1.0 - avgWeight) * nnUtility;
  }
  else if(searchParams.fpuParentWeight > 0.0) {
    parentUtilityForFPU = searchParams.fpuParentWeight * nnUtility + (1.0 - searchParams.fpuParentWeight) * parentUtility;
  }

  double fpuValue;
  {
    double fpuReductionMax = isRoot ? searchParams.rootFpuReductionMax : searchParams.fpuReductionMax;
    double fpuLossProp = isRoot ? searchParams.rootFpuLossProp : searchParams.fpuLossProp;
    double utilityRadius = searchParams.multiValueHeadUtilityMix == 0.0 ? searchParams.winLossUtilityFactor : 1.0;

    double reduction = fpuReductionMax * sqrt(policyProbMassVisited);
    fpuValue = pla == P_WHITE ? parentUtilityForFPU - reduction : parentUtilityForFPU + reduction;
    double lossValue = pla == P_WHITE ? -utilityRadius : utilityRadius;
    fpuValue = fpuValue + (lossValue - fpuValue) * fpuLossProp;
  }

  return fpuValue;
}


double Search::getFpuValueForChildrenAssumeVisited(
  const SearchNode& node, Player pla, bool isRoot, double policyProbMassVisited,
  double& parentUtility, double& parentWeightPerVisit, double& parentUtilityStdevFactor
) const {
  double weightSum = node.stats.weightSum.load(std::memory_order_acquire);
  double utilityAvg = node.stats.utilityAvg.load(std::memory_order_acquire);
  double utilitySqAvg = node.stats.utilitySqAvg.load(std::memory_order_acquire);
  return getFpuValueForChildrenAssumeVisitedByStats(
    node, pla, isRoot, policyProbMassVisited,
    node.stats.visits.load(std::memory_order_acquire),
    weightSum, utilityAvg, utilitySqAvg, getUtilityFromNN(*(node.getNNOutput()),node.nextPla),
    parentUtility, parentWeightPerVisit, parentUtilityStdevFactor
  );
}

void Search::selectBestVctChildToDescend(
  SearchThread& thread, const SearchNode& node, int nodeState,
  int& numChildrenFound, int& bestChildIdx, Loc& bestChildMoveLoc,
  bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE],
  bool isRoot) const
{
  Player attacker = thread.vctAttacker;
  assert(attacker == P_WHITE || attacker == P_BLACK);
  assert(thread.pla == node.nextPla);

  const NNOutput* nnOutput = node.getNNOutput();
  assert(nnOutput != NULL);
  if(!nnOutput->hasPolicyByHead())
    throw StringError("multi-head VCT search requires a v112 model with six policy heads");

  int vctPolicyHead = node.nextPla == attacker ? 4 : 5;
  int outcomePolicyHead = node.nextPla == attacker ? 3 : 2;
  double consensusMix = searchParams.multiHeadVctPolicyConsensusMix;
  double consensusMass = 0.0;
  if(consensusMix > 0.0) {
    for(int movePos = 0; movePos<policySize; movePos++) {
      if(nnOutput->getPolicyProbMaybeNoised(movePos) < 0.0f)
        continue;
      double vctPolicy = nnOutput->getPolicyProbByHead(vctPolicyHead,movePos);
      double outcomePolicy = nnOutput->getPolicyProbByHead(outcomePolicyHead,movePos);
      if(vctPolicy > 0.0 && outcomePolicy > 0.0)
        consensusMass += sqrt(vctPolicy * outcomePolicy);
    }
  }
  auto getVctPolicyProb = [&](int movePos) {
    float normalPolicy = nnOutput->getPolicyProbMaybeNoised(movePos);
    float vctPolicy = nnOutput->getPolicyProbByHead(vctPolicyHead,movePos);
    if(normalPolicy < 0.0f || vctPolicy < 0.0f)
      return -1.0f;
    double guidedPolicy = vctPolicy;
    if(consensusMix > 0.0 && consensusMass > 0.0) {
      double outcomePolicy =
        nnOutput->getPolicyProbByHead(outcomePolicyHead,movePos);
      double consensusPolicy =
        vctPolicy > 0.0 && outcomePolicy > 0.0 ?
        sqrt(vctPolicy * outcomePolicy) / consensusMass :
        0.0;
      guidedPolicy =
        (1.0 - consensusMix) * vctPolicy +
        consensusMix * consensusPolicy;
    }
    double mix = searchParams.multiHeadVctPolicyMix;
    return (float)((1.0 - mix) * normalPolicy + mix * guidedPolicy);
  };

  int childrenCapacity;
  const SearchChildPointer* children = node.getChildren(nodeState,childrenCapacity);
  if(
    isRoot &&
    thread.rootForcedReplySidecarMoveLoc != Board::NULL_LOC
  ) {
    numChildrenFound = 0;
    for(int i = 0; i<childrenCapacity; i++) {
      const SearchNode* child = children[i].getIfAllocated();
      if(child == NULL)
        break;
      numChildrenFound++;
      if(
        children[i].getMoveLocRelaxed() ==
        thread.rootForcedReplySidecarMoveLoc
      ) {
        bestChildIdx = i;
        bestChildMoveLoc = thread.rootForcedReplySidecarMoveLoc;
        return;
      }
    }
  }
  double policyProbMassVisited = 0.0;
  double totalChildWeight = 0.0;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    Loc moveLoc = children[i].getMoveLocRelaxed();
    float nnPolicyProb = getVctPolicyProb(getPos(moveLoc));
    if(nnPolicyProb < 0.0f)
      continue;
    policyProbMassVisited += nnPolicyProb;
    int64_t edgeVisits = children[i].getVctEdgeVisits(attacker);
    totalChildWeight += child->getVctStats(attacker).getChildWeight(edgeVisits);
  }
  if(policyProbMassVisited > 1.0)
    policyProbMassVisited = 1.0;

  VctStats parentStats(node.getVctStats(attacker));
  assert(parentStats.visits > 0);
  assert(parentStats.weightSum > 0.0);
  double parentWeightPerVisit = parentStats.weightSum / parentStats.visits;
  double parentUtility = parentStats.utilityAvg;

  double variancePrior = searchParams.cpuctUtilityStdevPrior * searchParams.cpuctUtilityStdevPrior;
  double variancePriorWeight = searchParams.cpuctUtilityStdevPriorWeight;
  double parentUtilityStdev;
  if(parentStats.weightSum <= 1.0)
    parentUtilityStdev = searchParams.cpuctUtilityStdevPrior;
  else {
    double utilitySq = parentUtility * parentUtility;
    double utilitySqAvg = std::max(parentStats.utilitySqAvg,utilitySq);
    parentUtilityStdev = sqrt(
      std::max(
        0.0,
        ((utilitySq + variancePrior) * variancePriorWeight + utilitySqAvg * parentStats.weightSum)
        / (variancePriorWeight + parentStats.weightSum - 1.0)
        - utilitySq
      )
    );
  }
  double parentUtilityStdevFactor =
    1.0 + searchParams.cpuctUtilityStdevScale *
    (parentUtilityStdev / searchParams.cpuctUtilityStdevPrior - 1.0);

  double nnUtility = searchParams.multiHeadVctUseNormalRules ?
    getUtilityFromNN(*nnOutput,node.nextPla) :
    getVctUtilityFromNN(*nnOutput,attacker,node.nextPla);
  double parentUtilityForFPU = parentUtility;
  if(searchParams.fpuParentWeightByVisitedPolicy) {
    double avgWeight = std::min(1.0,pow(policyProbMassVisited,searchParams.fpuParentWeightByVisitedPolicyPow));
    parentUtilityForFPU = avgWeight * parentUtility + (1.0 - avgWeight) * nnUtility;
  }
  else if(searchParams.fpuParentWeight > 0.0) {
    parentUtilityForFPU =
      searchParams.fpuParentWeight * nnUtility +
      (1.0 - searchParams.fpuParentWeight) * parentUtility;
  }

  double fpuReductionMax = isRoot ? searchParams.rootFpuReductionMax : searchParams.fpuReductionMax;
  double fpuLossProp = isRoot ? searchParams.rootFpuLossProp : searchParams.fpuLossProp;
  double reduction = fpuReductionMax * sqrt(policyProbMassVisited);
  double fpuValue = node.nextPla == P_WHITE ?
    parentUtilityForFPU - reduction :
    parentUtilityForFPU + reduction;
  double lossValue = node.nextPla == P_WHITE ? -1.0 : 1.0;
  fpuValue += (lossValue - fpuValue) * fpuLossProp;

  double exploreScaling =
    getExploreScaling(totalChildWeight,parentUtilityStdevFactor) *
    searchParams.multiHeadVctCpuctScale;

  std::fill(posesWithChildBuf,posesWithChildBuf+NNPos::MAX_NN_POLICY_SIZE,false);
  double maxSelectionValue = POLICY_ILLEGAL_SELECTION_VALUE;
  bestChildIdx = -1;
  bestChildMoveLoc = Board::NULL_LOC;
  numChildrenFound = 0;

  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    numChildrenFound++;

    Loc moveLoc = children[i].getMoveLocRelaxed();
    int movePos = getPos(moveLoc);
    posesWithChildBuf[movePos] = true;
    float nnPolicyProb = getVctPolicyProb(movePos);
    if(nnPolicyProb < 0.0f)
      continue;

    int64_t edgeVisits = children[i].getVctEdgeVisits(attacker);
    VctStats childStats(child->getVctStats(attacker));
    double childWeight = VctStats::childWeight(edgeVisits,childStats.visits,childStats.weightSum);
    double childUtility =
      childStats.visits <= 0 || childWeight <= 0.0 ?
      fpuValue : childStats.utilityAvg;

    int32_t childVirtualLosses = child->getVirtualLosses(attacker).load(std::memory_order_acquire);
    if(childVirtualLosses > 0) {
      double virtualLossWeight = childVirtualLosses * searchParams.numVirtualLossesPerThread;
      double virtualLossUtility = node.nextPla == P_WHITE ? -1.0 : 1.0;
      double virtualLossWeightFrac =
        virtualLossWeight / (virtualLossWeight + std::max(0.25,childWeight));
      childUtility += (virtualLossUtility - childUtility) * virtualLossWeightFrac;
      childWeight += virtualLossWeight;
    }

    double selectionValue = getExploreSelectionValue(
      exploreScaling,nnPolicyProb,childWeight,childUtility,node.nextPla
    );
    if(selectionValue > maxSelectionValue) {
      maxSelectionValue = selectionValue;
      bestChildIdx = i;
      bestChildMoveLoc = moveLoc;
    }
  }

  const std::vector<int>& avoidMoveUntilByLoc =
    thread.pla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
  Loc bestNewMoveLoc = Board::NULL_LOC;
  float bestNewNNPolicyProb = -1.0f;
  for(int movePos = 0; movePos<policySize; movePos++) {
    if(posesWithChildBuf[movePos])
      continue;
    Loc moveLoc = NNPos::posToLoc(movePos,thread.board.x_size,thread.board.y_size,nnXLen,nnYLen);
    if(moveLoc == Board::NULL_LOC)
      continue;
    if(isRoot && !isAllowedRootMove(moveLoc))
      continue;
    if(avoidMoveUntilByLoc.size() > 0) {
      int untilDepth = avoidMoveUntilByLoc[moveLoc];
      if(thread.history.moveHistory.size() - rootHistory.moveHistory.size() < untilDepth)
        continue;
    }

    float nnPolicyProb = getVctPolicyProb(movePos);
    if(nnPolicyProb > bestNewNNPolicyProb) {
      bestNewNNPolicyProb = nnPolicyProb;
      bestNewMoveLoc = moveLoc;
    }
  }

  if(bestNewMoveLoc != Board::NULL_LOC) {
    double selectionValue = getExploreSelectionValue(
      exploreScaling,bestNewNNPolicyProb,0.0,fpuValue,node.nextPla
    );
    if(selectionValue > maxSelectionValue) {
      bestChildIdx = numChildrenFound;
      bestChildMoveLoc = bestNewMoveLoc;
    }
  }
}

void Search::selectBestChildToDescend(
  SearchThread& thread, const SearchNode& node, int nodeState,
  int& numChildrenFound, int& bestChildIdx, Loc& bestChildMoveLoc,
  bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE],
  bool isRoot) const
{
  assert(thread.pla == node.nextPla);
  if(thread.vctAttacker != C_EMPTY) {
    selectBestVctChildToDescend(
      thread,node,nodeState,numChildrenFound,bestChildIdx,bestChildMoveLoc,
      posesWithChildBuf,isRoot
    );
    return;
  }
  if(
    isRoot &&
    thread.rootVctNormalVerificationMoveLoc != Board::NULL_LOC
  ) {
    int childrenCapacity;
    const SearchChildPointer* children = node.getChildren(nodeState,childrenCapacity);
    numChildrenFound = 0;
    for(int i = 0; i<childrenCapacity; i++) {
      const SearchNode* child = children[i].getIfAllocated();
      if(child == NULL)
        break;
      numChildrenFound++;
      if(
        children[i].getMoveLocRelaxed() ==
        thread.rootVctNormalVerificationMoveLoc
      ) {
        bestChildIdx = i;
        bestChildMoveLoc = thread.rootVctNormalVerificationMoveLoc;
        return;
      }
    }
  }

  double maxSelectionValue = POLICY_ILLEGAL_SELECTION_VALUE;
  bestChildIdx = -1;
  bestChildMoveLoc = Board::NULL_LOC;

  int childrenCapacity;
  const SearchChildPointer* children = node.getChildren(nodeState,childrenCapacity);

  double policyProbMassVisited = 0.0;
  double maxChildWeight = 0.0;
  double totalChildWeight = 0.0;
  double maxWhiteWinChildWeight = 0.0;
  double totalWhiteWinChildWeight = 0.0;
  double maxBlackWinChildWeight = 0.0;
  double totalBlackWinChildWeight = 0.0;
  const NNOutput* nnOutput = node.getNNOutput();
  assert(nnOutput != NULL);

  double multiHeadDrawIntervalSignal = 0.0;
  if(
    searchParams.multiHeadDrawPolicyFlattening > 0.0 ||
    searchParams.multiHeadDrawRootMinVisitsCoeff > 0.0 ||
    searchParams.multiHeadDrawAuxRootVisits > 0.0
  ) {
    if(!nnOutput->hasPolicyByHead())
      throw StringError("multi-head draw exploration requires a v112 model");
    double mustWinProb = node.nextPla == P_WHITE ?
      nnOutput->whiteWinProbByHead[3] :
      nnOutput->whiteLossProbByHead[3];
    double lossProbWhenDrawCountsAsWin = node.nextPla == P_WHITE ?
      nnOutput->whiteLossProbByHead[2] :
      nnOutput->whiteWinProbByHead[2];
    double nonLossProb = 1.0 - lossProbWhenDrawCountsAsWin;
    double outcomeInterval = std::clamp(
      nonLossProb - mustWinProb,
      0.0,1.0
    );
    multiHeadDrawIntervalSignal = sqrt(
      std::clamp((double)nnOutput->whiteNoResultProb,0.0,1.0) *
      outcomeInterval
    );
  }

  double normalVctPolicyMix = 0.0;
  double attackPolicyWeight = 0.0;
  double defensePolicyWeight = 0.0;
  double attackPolicyMix = 0.0;
  double defensePolicyMix = 0.0;
  double drawWinPolicyMix = 0.0;
  double drawLossPolicyMix = 0.0;
  double normalAuxPolicyMix = 0.0;
  bool useAuxPolicy =
    searchParams.multiHeadVctNormalPolicyMaxMix > 0.0 ||
    searchParams.multiHeadDrawWinNormalPolicyMaxMix > 0.0 ||
    searchParams.multiHeadDrawLossNormalPolicyMaxMix > 0.0 ||
    searchParams.multiHeadDrawPolicyFlattening > 0.0 ||
    searchParams.multiHeadDrawRootMinVisitsCoeff > 0.0 ||
    searchParams.multiHeadAuxPolicyOptimism > 0.0 ||
    searchParams.multiHeadAuxValueExplore > 0.0;
  if(useAuxPolicy) {
    if(!nnOutput->hasPolicyByHead())
      throw StringError("multi-head normal policy mixture requires a v112 model with six policy heads");
  }
  auto getPolicyWeight = [](double prob, double probScale, double exponent) {
    double poweredProb = pow(std::clamp(prob,0.0,1.0),exponent);
    double poweredScale = pow(probScale,exponent);
    return poweredProb / (poweredProb + poweredScale);
  };
  if(searchParams.multiHeadVctNormalPolicyMaxMix > 0.0) {
    double attackProb = node.nextPla == P_WHITE ?
      nnOutput->whiteWinProbByHead[4] :
      nnOutput->whiteLossProbByHead[4];
    double defenseProb = node.nextPla == P_WHITE ?
      nnOutput->whiteLossProbByHead[5] :
      nnOutput->whiteWinProbByHead[5];
    double attackShapedWeight = getPolicyWeight(
      attackProb,searchParams.multiHeadVctProbScale,searchParams.multiHeadVctNormalPolicyPow
    );
    double defenseShapedWeight = getPolicyWeight(
      defenseProb,searchParams.multiHeadVctProbScale,searchParams.multiHeadVctNormalPolicyPow
    );
    double rawProbMix = searchParams.multiHeadVctPolicyRawProbMix;
    attackPolicyWeight =
      searchParams.multiHeadVctAttackPolicyScale *
      (
        (1.0 - rawProbMix) * attackShapedWeight +
        rawProbMix * std::clamp(attackProb,0.0,1.0)
      );
    defensePolicyWeight =
      searchParams.multiHeadVctDefensePolicyScale *
      (
        (1.0 - rawProbMix) * defenseShapedWeight +
        rawProbMix * std::clamp(defenseProb,0.0,1.0)
      );
    attackPolicyMix =
      searchParams.multiHeadVctNormalPolicyMaxMix *
      attackPolicyWeight;
    defensePolicyMix =
      searchParams.multiHeadVctNormalPolicyMaxMix *
      defensePolicyWeight;
    double combinedWeight =
      1.0 -
      (1.0 - attackPolicyWeight) *
      (1.0 - defensePolicyWeight);
    normalVctPolicyMix = searchParams.multiHeadVctNormalPolicyMaxMix * combinedWeight;
  }
  if(
    searchParams.multiHeadDrawWinNormalPolicyMaxMix > 0.0 ||
    searchParams.multiHeadDrawLossNormalPolicyMaxMix > 0.0
  ) {
    double normalLossProb = node.nextPla == P_WHITE ?
      nnOutput->whiteLossProbByHead[0] :
      nnOutput->whiteWinProbByHead[0];
    double drawWinLossProb = node.nextPla == P_WHITE ?
      nnOutput->whiteLossProbByHead[2] :
      nnOutput->whiteWinProbByHead[2];
    double drawLossWinProb = node.nextPla == P_WHITE ?
      nnOutput->whiteWinProbByHead[3] :
      nnOutput->whiteLossProbByHead[3];
    double drawProb = nnOutput->whiteNoResultProb;

    //Head2 is most useful when normal play expects a loss but treating a draw
    //as a win reveals room to escape. Head3 is useful for both known wins and
    //drawish positions where normal policy may settle too early.
    double drawWinSignal =
      std::clamp(normalLossProb,0.0,1.0) * (1.0 - std::clamp(drawWinLossProb,0.0,1.0));
    double drawLossSignal =
      1.0 -
      (1.0 - std::clamp(drawLossWinProb,0.0,1.0)) *
      (1.0 - std::clamp(drawProb,0.0,1.0));
    double drawWinShapedWeight = getPolicyWeight(
      drawWinSignal,searchParams.multiHeadDrawNormalPolicyProbScale,searchParams.multiHeadDrawNormalPolicyPow
    );
    double drawLossShapedWeight = getPolicyWeight(
      drawLossSignal,searchParams.multiHeadDrawNormalPolicyProbScale,searchParams.multiHeadDrawNormalPolicyPow
    );
    double rawDrawProbMix = searchParams.multiHeadDrawPolicyRawProbMix;
    drawWinPolicyMix =
      searchParams.multiHeadDrawWinNormalPolicyMaxMix *
      (
        (1.0 - rawDrawProbMix) * drawWinShapedWeight +
        rawDrawProbMix * drawWinSignal
      );
    drawLossPolicyMix =
      searchParams.multiHeadDrawLossNormalPolicyMaxMix *
      (
        (1.0 - rawDrawProbMix) * drawLossShapedWeight +
        rawDrawProbMix * std::clamp(drawLossWinProb,0.0,1.0)
      );
  }
  normalAuxPolicyMix =
    1.0 -
    (1.0 - normalVctPolicyMix) *
    (1.0 - drawWinPolicyMix) *
    (1.0 - drawLossPolicyMix);

  double auxPolicyConcentration[NNOutput::NUM_POLICY_HEADS];
  std::fill(
    auxPolicyConcentration,
    auxPolicyConcentration+NNOutput::NUM_POLICY_HEADS,
    0.0
  );
  if(
    (
      searchParams.multiHeadAuxPolicyOptimism > 0.0 ||
      searchParams.multiHeadAuxValueExplore > 0.0
    ) &&
    normalAuxPolicyMix > 0.0
  ) {
    int legalPolicyCount = 0;
    for(int movePos = 0; movePos<policySize; movePos++) {
      if(nnOutput->getPolicyProbMaybeNoised(movePos) >= 0.0f)
        legalPolicyCount++;
    }
    double twiceUniformPolicy = legalPolicyCount > 0 ? 2.0 / legalPolicyCount : 1.0;
    for(int head = 2; head<NNOutput::NUM_POLICY_HEADS; head++) {
      double peakPolicy = 0.0;
      for(int movePos = 0; movePos<policySize; movePos++) {
        float policy = nnOutput->getPolicyProbByHead(head,movePos);
        if(policy > peakPolicy)
          peakPolicy = policy;
      }
      double peakExcess = std::max(0.0,peakPolicy - twiceUniformPolicy);
      auxPolicyConcentration[head] =
        peakExcess /
        (peakExcess + searchParams.multiHeadAuxPolicyConcentrationScale);
    }
  }

  const SearchNode* childByMovePos[NNPos::MAX_NN_POLICY_SIZE];
  int64_t edgeVisitsByMovePos[NNPos::MAX_NN_POLICY_SIZE];
  std::fill(childByMovePos,childByMovePos+NNPos::MAX_NN_POLICY_SIZE,nullptr);
  std::fill(edgeVisitsByMovePos,edgeVisitsByMovePos+NNPos::MAX_NN_POLICY_SIZE,0);
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    int movePos = getPos(children[i].getMoveLocRelaxed());
    childByMovePos[movePos] = child;
    edgeVisitsByMovePos[movePos] = children[i].getEdgeVisits();
  }

  double normalPolicyHead1Mix = searchParams.multiHeadNormalPolicyHead1Mix;
  if(normalPolicyHead1Mix > 0.0 && !nnOutput->hasPolicyByHead())
    throw StringError("head1 normal policy ensemble requires a v112 model");
  auto computeNormalSearchPolicyProb = [&](int movePos, const SearchNode* child) {
    float normalPolicy = nnOutput->getPolicyProbMaybeNoised(movePos);
    if(normalPolicyHead1Mix > 0.0) {
      float head1Policy = nnOutput->getPolicyProbByHead(1,movePos);
      if((normalPolicy < 0.0f) != (head1Policy < 0.0f))
        throw StringError("head0 and head1 policies disagree on legal moves");
      if(normalPolicy >= 0.0f) {
        normalPolicy = (float)(
          (1.0 - normalPolicyHead1Mix) * normalPolicy +
          normalPolicyHead1Mix * head1Policy
        );
      }
    }
    if(normalPolicy < 0.0f || normalAuxPolicyMix <= 0.0)
      return normalPolicy;

    double attackConfirmation = 1.0;
    double defenseConfirmation = 1.0;
    double drawWinConfirmation = 1.0;
    double drawLossConfirmation = 1.0;
    if(child != NULL && searchParams.multiHeadAuxPolicyChildGate > 0.0) {
      const NNOutput* childNNOutput = child->getNNOutput();
      if(childNNOutput != NULL) {
        Player parentPla = node.nextPla;
        Player opponentPla = getOpp(parentPla);
        double childGate = searchParams.multiHeadAuxPolicyChildGate;
        auto gatedConfirmation = [childGate](double prob) {
          return (1.0 - childGate) + childGate * std::clamp(prob,0.0,1.0);
        };

        double attackWinProb = parentPla == P_WHITE ?
          childNNOutput->whiteWinProbByHead[5] :
          childNNOutput->whiteLossProbByHead[5];
        double opponentAttackWinProb = opponentPla == P_WHITE ?
          childNNOutput->whiteWinProbByHead[4] :
          childNNOutput->whiteLossProbByHead[4];
        double drawLossWinProb = parentPla == P_WHITE ?
          childNNOutput->whiteWinProbByHead[2] :
          childNNOutput->whiteLossProbByHead[2];
        double drawWinOpponentWinProb = opponentPla == P_WHITE ?
          childNNOutput->whiteWinProbByHead[3] :
          childNNOutput->whiteLossProbByHead[3];

        attackConfirmation = gatedConfirmation(attackWinProb);
        defenseConfirmation = gatedConfirmation(1.0 - opponentAttackWinProb);
        drawLossConfirmation = gatedConfirmation(drawLossWinProb);
        drawWinConfirmation = gatedConfirmation(1.0 - drawWinOpponentWinProb);
      }
    }
    if(child != NULL && searchParams.multiHeadAuxPolicyVisitScale > 0.0) {
      double visitScale = searchParams.multiHeadAuxPolicyVisitScale;
      double visitDecay = visitScale / (visitScale + edgeVisitsByMovePos[movePos]);
      attackConfirmation *= visitDecay;
      defenseConfirmation *= visitDecay;
      drawWinConfirmation *= visitDecay;
      drawLossConfirmation *= visitDecay;
    }

    if(searchParams.multiHeadAuxPolicyOptimism > 0.0) {
      double optimisticPolicy = normalPolicy;
      auto applyOptimisticProposal = [&](int head, double policyMix, double confirmation) {
        if(policyMix <= 0.0)
          return true;
        float auxiliaryPolicy = nnOutput->getPolicyProbByHead(head,movePos);
        if(auxiliaryPolicy < 0.0f)
          return false;
        double proposalScale =
          searchParams.multiHeadAuxPolicyOptimism *
          policyMix *
          confirmation *
          auxPolicyConcentration[head];
        optimisticPolicy = std::max(
          optimisticPolicy,
          proposalScale * auxiliaryPolicy
        );
        return true;
      };
      if(!applyOptimisticProposal(4,attackPolicyMix,attackConfirmation))
        return -1.0f;
      if(!applyOptimisticProposal(5,defensePolicyMix,defenseConfirmation))
        return -1.0f;
      if(!applyOptimisticProposal(2,drawWinPolicyMix,drawWinConfirmation))
        return -1.0f;
      if(!applyOptimisticProposal(3,drawLossPolicyMix,drawLossConfirmation))
        return -1.0f;
      return (float)optimisticPolicy;
    }

    double auxPolicyWeightedSum = 0.0;
    double auxPolicyWeightSum = 0.0;
    double confirmedAuxWeightSum = 0.0;
    if(normalVctPolicyMix > 0.0) {
      float attackPolicy = nnOutput->getPolicyProbByHead(4,movePos);
      float defensePolicy = nnOutput->getPolicyProbByHead(5,movePos);
      if(attackPolicy < 0.0f || defensePolicy < 0.0f)
        return -1.0f;
      double tacticalWeight = attackPolicyWeight + defensePolicyWeight;
      double tacticalPolicy = tacticalWeight > 0.0 ?
        (
          attackPolicyWeight * attackConfirmation * attackPolicy +
          defensePolicyWeight * defenseConfirmation * defensePolicy
        ) / tacticalWeight :
        normalPolicy;
      double tacticalConfirmation = tacticalWeight > 0.0 ?
        (
          attackPolicyWeight * attackConfirmation +
          defensePolicyWeight * defenseConfirmation
        ) / tacticalWeight :
        1.0;
      auxPolicyWeightedSum += normalVctPolicyMix * tacticalPolicy;
      auxPolicyWeightSum += normalVctPolicyMix;
      confirmedAuxWeightSum += normalVctPolicyMix * tacticalConfirmation;
    }
    if(drawWinPolicyMix > 0.0) {
      float drawWinPolicy = nnOutput->getPolicyProbByHead(2,movePos);
      if(drawWinPolicy < 0.0f)
        return -1.0f;
      auxPolicyWeightedSum += drawWinPolicyMix * drawWinConfirmation * drawWinPolicy;
      auxPolicyWeightSum += drawWinPolicyMix;
      confirmedAuxWeightSum += drawWinPolicyMix * drawWinConfirmation;
    }
    if(drawLossPolicyMix > 0.0) {
      float drawLossPolicy = nnOutput->getPolicyProbByHead(3,movePos);
      if(drawLossPolicy < 0.0f)
        return -1.0f;
      auxPolicyWeightedSum += drawLossPolicyMix * drawLossConfirmation * drawLossPolicy;
      auxPolicyWeightSum += drawLossPolicyMix;
      confirmedAuxWeightSum += drawLossPolicyMix * drawLossConfirmation;
    }
    double auxPolicyScale = auxPolicyWeightSum > 0.0 ?
      normalAuxPolicyMix / auxPolicyWeightSum :
      0.0;
    double effectiveAuxPolicyMix = auxPolicyScale * confirmedAuxWeightSum;
    return (float)(
      (1.0 - effectiveAuxPolicyMix) * normalPolicy +
      auxPolicyScale * auxPolicyWeightedSum
    );
  };

  float normalSearchPolicyProbs[NNPos::MAX_NN_POLICY_SIZE];
  std::fill(
    normalSearchPolicyProbs,
    normalSearchPolicyProbs+NNPos::MAX_NN_POLICY_SIZE,
    -1.0f
  );
  double normalSearchPolicyMass = 0.0;
  for(int movePos = 0; movePos<policySize; movePos++) {
    float policyProb = computeNormalSearchPolicyProb(movePos,childByMovePos[movePos]);
    normalSearchPolicyProbs[movePos] = policyProb;
    if(policyProb >= 0.0f)
      normalSearchPolicyMass += policyProb;
  }
  if(
    (
      searchParams.multiHeadAuxPolicyChildGate > 0.0 ||
      searchParams.multiHeadAuxPolicyVisitScale > 0.0 ||
      searchParams.multiHeadAuxPolicyOptimism > 0.0
    ) &&
    normalAuxPolicyMix > 0.0 &&
    normalSearchPolicyMass > 0.0
  ) {
    for(int movePos = 0; movePos<policySize; movePos++) {
      if(normalSearchPolicyProbs[movePos] >= 0.0f)
        normalSearchPolicyProbs[movePos] = (float)(normalSearchPolicyProbs[movePos] / normalSearchPolicyMass);
    }
  }
  if(searchParams.multiHeadDrawPolicyFlattening > 0.0) {
    double flattening =
      searchParams.multiHeadDrawPolicyFlattening *
      multiHeadDrawIntervalSignal;
    double policyExponent = 1.0 - 0.5 * flattening;
    if(policyExponent < 1.0) {
      double flattenedMass = 0.0;
      for(int movePos = 0; movePos<policySize; movePos++) {
        float policyProb = normalSearchPolicyProbs[movePos];
        if(policyProb <= 0.0f)
          continue;
        normalSearchPolicyProbs[movePos] =
          (float)pow((double)policyProb,policyExponent);
        flattenedMass += normalSearchPolicyProbs[movePos];
      }
      if(flattenedMass > 0.0) {
        for(int movePos = 0; movePos<policySize; movePos++) {
          if(normalSearchPolicyProbs[movePos] >= 0.0f)
            normalSearchPolicyProbs[movePos] =
              (float)(normalSearchPolicyProbs[movePos] / flattenedMass);
        }
      }
    }
  }
  if(thread.policyGuidanceMode != SearchThread::POLICY_GUIDANCE_NONE) {
    if(thread.policyGuidanceAttacker != P_WHITE && thread.policyGuidanceAttacker != P_BLACK)
      throw StringError("guided playout has no fixed attacker");
    if(!nnOutput->hasPolicyByHead())
      throw StringError("multi-head guided playouts require a v112 model with six policy heads");

    bool attackerToMove = node.nextPla == thread.policyGuidanceAttacker;
    int guidedHead;
    if(thread.policyGuidanceMode == SearchThread::POLICY_GUIDANCE_VCT)
      guidedHead = attackerToMove ? 4 : 5;
    else if(thread.policyGuidanceMode == SearchThread::POLICY_GUIDANCE_MUST_WIN)
      guidedHead = attackerToMove ? 3 : 2;
    else
      throw StringError("invalid guided playout mode");

    double guidedPolicyMix = searchParams.multiHeadGuidedPolicyMix;
    for(int movePos = 0; movePos<policySize; movePos++) {
      float normalPolicy = normalSearchPolicyProbs[movePos];
      if(normalPolicy < 0.0f)
        continue;
      float guidedPolicy = nnOutput->getPolicyProbByHead(guidedHead,movePos);
      if(guidedPolicy < 0.0f)
        throw StringError("guided and normal policy heads disagree on legal moves");
      normalSearchPolicyProbs[movePos] = (float)(
        (1.0 - guidedPolicyMix) * normalPolicy +
        guidedPolicyMix * guidedPolicy
      );
    }
  }
  float winObjectivePolicyProbs[NNPos::MAX_NN_POLICY_SIZE];
  float nonLossObjectivePolicyProbs[NNPos::MAX_NN_POLICY_SIZE];
  std::fill(
    winObjectivePolicyProbs,
    winObjectivePolicyProbs+NNPos::MAX_NN_POLICY_SIZE,
    -1.0f
  );
  std::fill(
    nonLossObjectivePolicyProbs,
    nonLossObjectivePolicyProbs+NNPos::MAX_NN_POLICY_SIZE,
    -1.0f
  );
  double objectivePolicyMix = searchParams.multiHeadObjectivePolicyMix;
  if(objectivePolicyMix > 0.0 && !nnOutput->hasPolicyByHead())
    throw StringError("multi-head objective policies require a v112 model with six policy heads");
  for(int movePos = 0; movePos<policySize; movePos++) {
    float normalPolicy = normalSearchPolicyProbs[movePos];
    if(normalPolicy < 0.0f) {
      winObjectivePolicyProbs[movePos] = normalPolicy;
      nonLossObjectivePolicyProbs[movePos] = normalPolicy;
      continue;
    }
    if(objectivePolicyMix <= 0.0) {
      winObjectivePolicyProbs[movePos] = normalPolicy;
      nonLossObjectivePolicyProbs[movePos] = normalPolicy;
      continue;
    }
    float winPolicy = nnOutput->getPolicyProbByHead(3,movePos);
    float nonLossPolicy = nnOutput->getPolicyProbByHead(2,movePos);
    if(winPolicy < 0.0f || nonLossPolicy < 0.0f)
      throw StringError("objective and normal policy heads disagree on legal moves");
    winObjectivePolicyProbs[movePos] = (float)(
      (1.0 - objectivePolicyMix) * normalPolicy +
      objectivePolicyMix * winPolicy
    );
    nonLossObjectivePolicyProbs[movePos] = (float)(
      (1.0 - objectivePolicyMix) * normalPolicy +
      objectivePolicyMix * nonLossPolicy
    );
  }
  auto getNormalSearchPolicyProb = [&](int movePos) {
    return normalSearchPolicyProbs[movePos];
  };
  auto getWhiteWinSearchPolicyProb = [&](int movePos) {
    return node.nextPla == P_WHITE ?
      winObjectivePolicyProbs[movePos] :
      nonLossObjectivePolicyProbs[movePos];
  };
  auto getBlackWinSearchPolicyProb = [&](int movePos) {
    return node.nextPla == P_BLACK ?
      winObjectivePolicyProbs[movePos] :
      nonLossObjectivePolicyProbs[movePos];
  };

  auto getGuidedValueSelectionAdjustment = [&](const SearchNode* child) {
    if(
      thread.policyGuidanceMode == SearchThread::POLICY_GUIDANCE_NONE ||
      searchParams.multiHeadGuidedValueWeight <= 0.0
    )
      return 0.0;
    const NNOutput* childNNOutput = child->getNNOutput();
    if(childNNOutput == NULL)
      return 0.0;

    Player attacker = thread.policyGuidanceAttacker;
    if(attacker != P_WHITE && attacker != P_BLACK)
      throw StringError("guided playout has no fixed attacker");
    double guidedUtilityWhite;
    if(thread.policyGuidanceMode == SearchThread::POLICY_GUIDANCE_VCT)
      guidedUtilityWhite = getVctUtilityFromNN(*childNNOutput,attacker,child->nextPla);
    else if(thread.policyGuidanceMode == SearchThread::POLICY_GUIDANCE_MUST_WIN) {
      int head = child->nextPla == attacker ? 3 : 2;
      double attackerWinProb = attacker == P_WHITE ?
        childNNOutput->whiteWinProbByHead[head] :
        childNNOutput->whiteLossProbByHead[head];
      guidedUtilityWhite = attacker == P_WHITE ?
        2.0 * attackerWinProb - 1.0 :
        1.0 - 2.0 * attackerWinProb;
    }
    else
      throw StringError("invalid guided playout mode");

    double normalUtilityWhite = getUtilityFromNN(*childNNOutput,child->nextPla);
    double utilityResidualWhite = guidedUtilityWhite - normalUtilityWhite;
    double utilityResidualSelf =
      node.nextPla == P_WHITE ? utilityResidualWhite : -utilityResidualWhite;
    return searchParams.multiHeadGuidedValueWeight * utilityResidualSelf;
  };

  auto getOutcomeIntervalExploreBonus = [&](
    const SearchNode* child, int64_t childEdgeVisits
  ) {
    if(searchParams.multiHeadOutcomeIntervalExplore <= 0.0)
      return 0.0;
    const NNOutput* childNNOutput = child->getNNOutput();
    if(childNNOutput == NULL)
      return 0.0;
    if(!childNNOutput->hasPolicyByHead())
      throw StringError("multi-head outcome-interval exploration requires a v112 model");

    Player parentPla = node.nextPla;
    Player childPla = child->nextPla;
    double robustParentWinProb = parentPla == P_WHITE ?
      childNNOutput->whiteWinProbByHead[2] :
      childNNOutput->whiteLossProbByHead[2];
    double robustChildWinProb = childPla == P_WHITE ?
      childNNOutput->whiteWinProbByHead[3] :
      childNNOutput->whiteLossProbByHead[3];
    robustParentWinProb = std::clamp(robustParentWinProb,0.0,1.0);
    double upperParentWinProb = std::clamp(
      1.0 - robustChildWinProb,
      robustParentWinProb,1.0
    );

    //Suppress uncertainty that head0 already identifies as a stable draw, but
    //retain the part of the interval that can still become a decisive result.
    double decisiveProb =
      1.0 - std::clamp((double)childNNOutput->whiteNoResultProb,0.0,1.0);
    double targetParentWinProb =
      robustParentWinProb +
      0.5 * (upperParentWinProb - robustParentWinProb) * decisiveProb;
    double normalParentWinProb = parentPla == P_WHITE ?
      childNNOutput->whiteWinProbByHead[0] :
      childNNOutput->whiteLossProbByHead[0];
    double winProbGain = std::max(
      0.0,
      targetParentWinProb - normalParentWinProb
    );
    double visitScale = searchParams.multiHeadOutcomeIntervalVisitScale;
    double visitDecay = visitScale / (visitScale + childEdgeVisits);
    return
      2.0 *
      searchParams.multiHeadOutcomeIntervalExplore *
      visitDecay *
      winProbGain;
  };

  auto getAuxiliaryValueExploreBonus = [&](
    const SearchNode* child, int movePos, int64_t childEdgeVisits
  ) {
    if(searchParams.multiHeadAuxValueExplore <= 0.0 || normalAuxPolicyMix <= 0.0)
      return 0.0;
    const NNOutput* childNNOutput = child->getNNOutput();
    if(childNNOutput == NULL)
      return 0.0;

    int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
    double childWeight = child->stats.weightSum.load(std::memory_order_acquire);
    if(childVisits <= 0 || childWeight <= 0.0)
      return 0.0;

    double childUtilityWhite = child->stats.utilityAvg.load(std::memory_order_acquire);
    double childNoResult = child->stats.noResultValueAvg.load(std::memory_order_acquire);
    double parentNoResult = node.stats.noResultValueAvg.load(std::memory_order_acquire);
    double noResultReduce = searchParams.noResultUtilityReduce * (1.0 - parentNoResult);
    childUtilityWhite -=
      childNoResult *
      noResultUtilityDecrease(
        searchParams.noResultUtilityForWhite,noResultReduce,node.nextPla
      );
    double normalSelfUtility =
      node.nextPla == P_WHITE ? childUtilityWhite : -childUtilityWhite;

    float normalPolicy = nnOutput->getPolicyProbMaybeNoised(movePos);
    if(normalPolicy < 0.0f)
      return 0.0;
    auto getProposalSupport = [&](int head, double policyMix) {
      if(policyMix <= 0.0)
        return 0.0;
      float auxiliaryPolicy = nnOutput->getPolicyProbByHead(head,movePos);
      if(auxiliaryPolicy <= 0.0f)
        return 0.0;
      double proposal =
        policyMix *
        auxPolicyConcentration[head] *
        auxiliaryPolicy;
      if(proposal <= normalPolicy)
        return 0.0;
      return 1.0 - normalPolicy / proposal;
    };

    Player parentPla = node.nextPla;
    Player opponentPla = getOpp(parentPla);
    auto winProbFor = [](const NNOutput& output, int head, Player pla) {
      return pla == P_WHITE ?
        (double)output.whiteWinProbByHead[head] :
        (double)output.whiteLossProbByHead[head];
    };

    double bestBonus = 0.0;
    auto addOptimisticTarget = [&](double support, double targetSelfUtility) {
      if(support <= 0.0)
        return;
      bestBonus = std::max(
        bestBonus,
        support * std::max(0.0,targetSelfUtility - normalSelfUtility)
      );
    };

    //After the parent moves, head5 evaluates the parent's fixed VCT attack,
    //while head2 evaluates the parent's draw-as-loss objective.
    addOptimisticTarget(
      getProposalSupport(4,attackPolicyMix),
      2.0 * winProbFor(*childNNOutput,5,parentPla) - 1.0
    );
    addOptimisticTarget(
      getProposalSupport(3,drawLossPolicyMix),
      2.0 * winProbFor(*childNNOutput,2,parentPla) - 1.0
    );

    //For defense and draw-escape proposals, the child is the attacker. Convert
    //its win probability to the parent's white-positive "not losing" utility.
    addOptimisticTarget(
      getProposalSupport(5,defensePolicyMix),
      1.0 - 2.0 * winProbFor(*childNNOutput,4,opponentPla)
    );
    addOptimisticTarget(
      getProposalSupport(2,drawWinPolicyMix),
      1.0 - 2.0 * winProbFor(*childNNOutput,3,opponentPla)
    );

    double visitScale = searchParams.multiHeadAuxValueVisitScale;
    double visitDecay = visitScale / (visitScale + childEdgeVisits);
    return searchParams.multiHeadAuxValueExplore * visitDecay * bestBonus;
  };

  bool separateObjectivePlayouts =
    searchParams.multiHeadObjectiveSeparatePlayouts;
  double whiteWinPolicyProbMassVisited = 0.0;
  double blackWinPolicyProbMassVisited = 0.0;
  double whiteWinObjectiveVisits = 1.0;
  double blackWinObjectiveVisits = 1.0;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    Loc moveLoc = children[i].getMoveLocRelaxed();
    int movePos = getPos(moveLoc);
    float nnPolicyProb = getNormalSearchPolicyProb(movePos);
    if(nnPolicyProb < 0)
      continue;
    policyProbMassVisited += nnPolicyProb;

    int64_t edgeVisits = children[i].getEdgeVisits();
    int64_t whiteWinEdgeVisits =
      children[i].getObjectiveEdgeVisits(P_WHITE);
    int64_t blackWinEdgeVisits =
      children[i].getObjectiveEdgeVisits(P_BLACK);
    double childWeight = child->stats.getChildWeight(edgeVisits);
    double crossWeight = searchParams.multiHeadObjectiveCrossWeight;
    double whiteWinCreditedVisits =
      whiteWinEdgeVisits + crossWeight * blackWinEdgeVisits;
    double blackWinCreditedVisits =
      blackWinEdgeVisits + crossWeight * whiteWinEdgeVisits;
    double whiteWinChildWeight = separateObjectivePlayouts ?
      whiteWinCreditedVisits :
      child->stats.getChildWhiteWinWeight(edgeVisits);
    double blackWinChildWeight = separateObjectivePlayouts ?
      blackWinCreditedVisits :
      child->stats.getChildBlackWinWeight(edgeVisits);
    if(!separateObjectivePlayouts || whiteWinCreditedVisits > 0.0)
      whiteWinPolicyProbMassVisited += getWhiteWinSearchPolicyProb(movePos);
    if(!separateObjectivePlayouts || blackWinCreditedVisits > 0.0)
      blackWinPolicyProbMassVisited += getBlackWinSearchPolicyProb(movePos);
    if(separateObjectivePlayouts) {
      whiteWinObjectiveVisits += whiteWinCreditedVisits;
      blackWinObjectiveVisits += blackWinCreditedVisits;
    }

    totalChildWeight += childWeight;
    if(childWeight > maxChildWeight)
      maxChildWeight = childWeight;
    totalWhiteWinChildWeight += whiteWinChildWeight;
    if(whiteWinChildWeight > maxWhiteWinChildWeight)
      maxWhiteWinChildWeight = whiteWinChildWeight;
    totalBlackWinChildWeight += blackWinChildWeight;
    if(blackWinChildWeight > maxBlackWinChildWeight)
      maxBlackWinChildWeight = blackWinChildWeight;
  }
  //Probability mass should not sum to more than 1, giving a generous allowance
  //for floating point error.
  assert(policyProbMassVisited <= 1.1);
  if(policyProbMassVisited > 1.0)
    policyProbMassVisited = 1.0;
  assert(whiteWinPolicyProbMassVisited <= 1.1);
  if(whiteWinPolicyProbMassVisited > 1.0)
    whiteWinPolicyProbMassVisited = 1.0;
  assert(blackWinPolicyProbMassVisited <= 1.1);
  if(blackWinPolicyProbMassVisited > 1.0)
    blackWinPolicyProbMassVisited = 1.0;

  //First play urgency
  double parentUtility;
  double parentWeightPerVisit;
  double parentUtilityStdevFactor;
  double fpuValue = getFpuValueForChildrenAssumeVisited(
    node, thread.pla, isRoot, policyProbMassVisited,
    parentUtility, parentWeightPerVisit, parentUtilityStdevFactor
  );

  bool useMultiValueHeads = searchParams.multiValueHeadUtilityMix != 0.0;
  double whiteWinParentUtility = parentUtility;
  double whiteWinParentWeightPerVisit = parentWeightPerVisit;
  double whiteWinParentUtilityStdevFactor = parentUtilityStdevFactor;
  double whiteWinFpuValue = fpuValue;
  double blackWinParentUtility = parentUtility;
  double blackWinParentWeightPerVisit = parentWeightPerVisit;
  double blackWinParentUtilityStdevFactor = parentUtilityStdevFactor;
  double blackWinFpuValue = fpuValue;
  if(useMultiValueHeads) {
    double whiteWinWeightSum = node.stats.whiteWinWeightSum.load(std::memory_order_acquire);
    double whiteWinUtilityAvg = node.stats.whiteWinUtilityAvg.load(std::memory_order_acquire);
    double whiteWinUtilitySqAvg = node.stats.whiteWinUtilitySqAvg.load(std::memory_order_acquire);
    if(whiteWinWeightSum > 0.0) {
      whiteWinFpuValue = getFpuValueForChildrenAssumeVisitedByStats(
        node, thread.pla, isRoot, whiteWinPolicyProbMassVisited,
        separateObjectivePlayouts ?
          whiteWinObjectiveVisits :
          node.stats.visits.load(std::memory_order_acquire),
        whiteWinWeightSum, whiteWinUtilityAvg, whiteWinUtilitySqAvg, getWhiteWinUtilityFromNN(*nnOutput,node.nextPla),
        whiteWinParentUtility, whiteWinParentWeightPerVisit, whiteWinParentUtilityStdevFactor
      );
    }

    double blackWinWeightSum = node.stats.blackWinWeightSum.load(std::memory_order_acquire);
    double blackWinUtilityInvAvg = node.stats.blackWinUtilityInvAvg.load(std::memory_order_acquire);
    double blackWinUtilityInvSqAvg = node.stats.blackWinUtilityInvSqAvg.load(std::memory_order_acquire);
    if(blackWinWeightSum > 0.0) {
      blackWinFpuValue = getFpuValueForChildrenAssumeVisitedByStats(
        node, thread.pla, isRoot, blackWinPolicyProbMassVisited,
        separateObjectivePlayouts ?
          blackWinObjectiveVisits :
          node.stats.visits.load(std::memory_order_acquire),
        blackWinWeightSum, blackWinUtilityInvAvg, blackWinUtilityInvSqAvg, getBlackWinUtilityInvFromNN(*nnOutput,node.nextPla),
        blackWinParentUtility, blackWinParentWeightPerVisit, blackWinParentUtilityStdevFactor
      );
    }
  }

  std::fill(posesWithChildBuf,posesWithChildBuf+NNPos::MAX_NN_POLICY_SIZE,false);

  double exploreScaling = getExploreScaling(totalChildWeight, parentUtilityStdevFactor);
  double whiteWinExploreScaling = getExploreScaling(totalWhiteWinChildWeight, whiteWinParentUtilityStdevFactor);
  double blackWinExploreScaling = getExploreScaling(totalBlackWinChildWeight, blackWinParentUtilityStdevFactor);

  double forcedReplyAuxPolicyScale[NNPos::MAX_NN_POLICY_SIZE];
  std::fill(
    forcedReplyAuxPolicyScale,
    forcedReplyAuxPolicyScale+NNPos::MAX_NN_POLICY_SIZE,
    1.0
  );
  double forcedReplyAuxPolicyGate =
    searchParams.multiHeadDrawForcedReplyAuxPolicyGate;
  if(isRoot && forcedReplyAuxPolicyGate > 0.0) {
    if(!nnOutput->hasPolicyByHead())
      throw StringError("multi-head forced-reply policy gate requires a v112 model");
    double sideToMoveWinWeight =
      getSideToMoveWinObjectiveWeight(node);
    double maxConsensus = 0.0;
    for(int movePos = 0; movePos<policySize; movePos++) {
      if(normalSearchPolicyProbs[movePos] < 0.0f) {
        forcedReplyAuxPolicyScale[movePos] = 0.0;
        continue;
      }
      double attackConsensus = sqrt(
        std::max(0.0,(double)nnOutput->getPolicyProbByHead(3,movePos)) *
        std::max(0.0,(double)nnOutput->getPolicyProbByHead(4,movePos))
      );
      double defenseConsensus = sqrt(
        std::max(0.0,(double)nnOutput->getPolicyProbByHead(2,movePos)) *
        std::max(0.0,(double)nnOutput->getPolicyProbByHead(5,movePos))
      );
      double consensus =
        sideToMoveWinWeight * attackConsensus +
        (1.0 - sideToMoveWinWeight) * defenseConsensus;
      forcedReplyAuxPolicyScale[movePos] = consensus;
      maxConsensus = std::max(maxConsensus,consensus);
    }
    if(maxConsensus > 0.0) {
      for(int movePos = 0; movePos<policySize; movePos++) {
        double relativeConsensus =
          forcedReplyAuxPolicyScale[movePos] / maxConsensus;
        forcedReplyAuxPolicyScale[movePos] =
          (1.0 - forcedReplyAuxPolicyGate) +
          forcedReplyAuxPolicyGate * relativeConsensus;
      }
    }
    else {
      std::fill(
        forcedReplyAuxPolicyScale,
        forcedReplyAuxPolicyScale+NNPos::MAX_NN_POLICY_SIZE,
        1.0
      );
    }
  }

  double rootAuxDesiredVisits[NNPos::MAX_NN_POLICY_SIZE];
  std::fill(
    rootAuxDesiredVisits,
    rootAuxDesiredVisits+NNPos::MAX_NN_POLICY_SIZE,
    0.0
  );
  if(isRoot && searchParams.multiHeadDrawAuxRootVisits > 0.0) {
    if(!nnOutput->hasPolicyByHead())
      throw StringError("multi-head root auxiliary proposals require a v112 model");
    double sideToMoveWinWeight =
      getSideToMoveWinObjectiveWeight(node);
    double maxConsensus = 0.0;
    for(int movePos = 0; movePos<policySize; movePos++) {
      if(normalSearchPolicyProbs[movePos] < 0.0f)
        continue;
      double attackConsensus = sqrt(
        std::max(0.0,(double)nnOutput->getPolicyProbByHead(3,movePos)) *
        std::max(0.0,(double)nnOutput->getPolicyProbByHead(4,movePos))
      );
      double defenseConsensus = sqrt(
        std::max(0.0,(double)nnOutput->getPolicyProbByHead(2,movePos)) *
        std::max(0.0,(double)nnOutput->getPolicyProbByHead(5,movePos))
      );
      double consensus =
        sideToMoveWinWeight * attackConsensus +
        (1.0 - sideToMoveWinWeight) * defenseConsensus;
      rootAuxDesiredVisits[movePos] = consensus;
      maxConsensus = std::max(maxConsensus,consensus);
    }
    if(maxConsensus > 0.0) {
      for(int movePos = 0; movePos<policySize; movePos++) {
        double relativeConsensus =
          rootAuxDesiredVisits[movePos] / maxConsensus;
        rootAuxDesiredVisits[movePos] =
          searchParams.multiHeadDrawAuxRootVisits *
          multiHeadDrawIntervalSignal *
          relativeConsensus * relativeConsensus;
      }
    }
  }
  if(isRoot && searchParams.multiHeadTacticalRootVisits > 0.0) {
    if(!nnOutput->hasPolicyByHead())
      throw StringError("multi-head tactical root proposals require a v112 model");

    int legalPolicyCount = 0;
    for(int movePos = 0; movePos<policySize; movePos++) {
      if(normalSearchPolicyProbs[movePos] >= 0.0f)
        legalPolicyCount++;
    }
    double uniformPolicy =
      legalPolicyCount > 0 ? 1.0 / legalPolicyCount : 1.0;

    double tacticalNormalSelfUtility[NNPos::MAX_NN_POLICY_SIZE];
    int64_t tacticalNormalEdgeVisits[NNPos::MAX_NN_POLICY_SIZE];
    std::fill(
      tacticalNormalSelfUtility,
      tacticalNormalSelfUtility+NNPos::MAX_NN_POLICY_SIZE,
      0.0
    );
    std::fill(
      tacticalNormalEdgeVisits,
      tacticalNormalEdgeVisits+NNPos::MAX_NN_POLICY_SIZE,
      0
    );
    double bestTacticalNormalSelfUtility = -1e20;
    if(searchParams.multiHeadTacticalDisproofStrength > 0.0) {
      for(int i = 0; i<childrenCapacity; i++) {
        const SearchNode* child = children[i].getIfAllocated();
        if(child == NULL)
          break;
        int64_t edgeVisits = children[i].getEdgeVisits();
        int64_t childVisits =
          child->stats.visits.load(std::memory_order_acquire);
        double childWeight =
          child->stats.getChildWeight(edgeVisits,childVisits);
        if(edgeVisits <= 0 || childVisits <= 0 || childWeight <= 0.0)
          continue;

        double childUtility =
          child->stats.utilityAvg.load(std::memory_order_acquire);
        double childNoResult =
          child->stats.noResultValueAvg.load(std::memory_order_acquire);
        double parentNoResult =
          node.stats.noResultValueAvg.load(std::memory_order_acquire);
        double noResultReduce =
          searchParams.noResultUtilityReduce * (1.0 - parentNoResult);
        childUtility -=
          childNoResult *
          noResultUtilityDecrease(
            searchParams.noResultUtilityForWhite,
            noResultReduce,
            node.nextPla
          );
        double childSelfUtility =
          node.nextPla == P_WHITE ? childUtility : -childUtility;
        int movePos = getPos(children[i].getMoveLocRelaxed());
        tacticalNormalSelfUtility[movePos] = childSelfUtility;
        tacticalNormalEdgeVisits[movePos] = edgeVisits;
        bestTacticalNormalSelfUtility =
          std::max(bestTacticalNormalSelfUtility,childSelfUtility);
      }
    }

    double peakPolicy[NNOutput::NUM_POLICY_HEADS];
    double concentration[NNOutput::NUM_POLICY_HEADS];
    double contrastiveConcentration[NNOutput::NUM_POLICY_HEADS];
    std::fill(
      peakPolicy,
      peakPolicy+NNOutput::NUM_POLICY_HEADS,
      0.0
    );
    std::fill(
      concentration,
      concentration+NNOutput::NUM_POLICY_HEADS,
      0.0
    );
    std::fill(
      contrastiveConcentration,
      contrastiveConcentration+NNOutput::NUM_POLICY_HEADS,
      0.0
    );
    for(int head = 2; head<NNOutput::NUM_POLICY_HEADS; head++) {
      for(int movePos = 0; movePos<policySize; movePos++) {
        if(normalSearchPolicyProbs[movePos] < 0.0f)
          continue;
        peakPolicy[head] = std::max(
          peakPolicy[head],
          (double)nnOutput->getPolicyProbByHead(head,movePos)
        );
      }
      double peakExcess =
        std::max(0.0,peakPolicy[head] - 2.0 * uniformPolicy);
      concentration[head] =
        peakExcess /
        (
          peakExcess +
          searchParams.multiHeadAuxPolicyConcentrationScale
        );
      contrastiveConcentration[head] =
        peakPolicy[head] /
        (
          peakPolicy[head] +
          searchParams.multiHeadAuxPolicyConcentrationScale
        );
    }

    auto sideWinProb = [&](int head) {
      return node.nextPla == P_WHITE ?
        (double)nnOutput->whiteWinProbByHead[head] :
        (double)nnOutput->whiteLossProbByHead[head];
    };
    auto sideLossProb = [&](int head) {
      return node.nextPla == P_WHITE ?
        (double)nnOutput->whiteLossProbByHead[head] :
        (double)nnOutput->whiteWinProbByHead[head];
    };
    auto smoothOpportunity = [&](double probability) {
      probability = std::clamp(probability,0.0,1.0);
      return
        probability /
        (probability + searchParams.multiHeadVctProbScale);
    };

    double mustWinProb = sideWinProb(3);
    double nonLossProb = 1.0 - sideLossProb(2);
    double objectiveOpportunity[NNOutput::NUM_POLICY_HEADS];
    std::fill(
      objectiveOpportunity,
      objectiveOpportunity+NNOutput::NUM_POLICY_HEADS,
      0.0
    );
    objectiveOpportunity[2] = smoothOpportunity(1.0 - nonLossProb);
    objectiveOpportunity[3] = smoothOpportunity(
      nonLossProb * (1.0 - mustWinProb)
    );
    objectiveOpportunity[4] =
      smoothOpportunity(sideWinProb(4)) *
      (1.0 - pow(mustWinProb,8.0));
    objectiveOpportunity[5] = smoothOpportunity(sideLossProb(5));

    double contrastiveMix =
      searchParams.multiHeadTacticalContrastiveMix;
    for(int movePos = 0; movePos<policySize; movePos++) {
      if(normalSearchPolicyProbs[movePos] < 0.0f)
        continue;
      double normalPolicy =
        0.5 *
        (
          std::max(
            0.0,
            (double)nnOutput->getPolicyProbByHead(0,movePos)
          ) +
          std::max(
            0.0,
            (double)nnOutput->getPolicyProbByHead(1,movePos)
          )
        );
      double bestProposal = 0.0;
      for(int head = 2; head<NNOutput::NUM_POLICY_HEADS; head++) {
        if(
          (
            searchParams.multiHeadTacticalObjectiveMask &
            (1 << (head - 2))
          ) == 0
        )
          continue;
        if(
          peakPolicy[head] <= 0.0 ||
          objectiveOpportunity[head] <= 0.0
        )
          continue;
        double auxiliaryPolicy = std::max(
          0.0,
          (double)nnOutput->getPolicyProbByHead(head,movePos)
        );
        double relativePolicy =
          auxiliaryPolicy /
          peakPolicy[head];
        double relativeLift = std::clamp(
          (auxiliaryPolicy - normalPolicy) /
            (
              auxiliaryPolicy +
              normalPolicy +
              0.02 * uniformPolicy
            ),
          0.0,1.0
        );
        double concentratedRelativePolicy = pow(
          relativePolicy,
          searchParams.multiHeadTacticalPolicyPower
        );
        double policyProposal =
          (1.0 - contrastiveMix) *
            concentratedRelativePolicy *
            concentratedRelativePolicy +
          contrastiveMix *
            concentratedRelativePolicy * relativeLift;
        double policyTrust =
          (1.0 - contrastiveMix) * concentration[head] +
          contrastiveMix * contrastiveConcentration[head];
        double proposal =
          objectiveOpportunity[head] *
          policyTrust *
          policyProposal;
        bestProposal = std::max(bestProposal,proposal);
      }
      int64_t normalEdgeVisits =
        tacticalNormalEdgeVisits[movePos];
      if(
        searchParams.multiHeadTacticalDisproofStrength > 0.0 &&
        normalEdgeVisits > 0 &&
        bestTacticalNormalSelfUtility > -1e10
      ) {
        double optimisticSelfUtility =
          tacticalNormalSelfUtility[movePos] +
          0.5 / sqrt((double)normalEdgeVisits);
        double utilityGap = std::max(
          0.0,
          bestTacticalNormalSelfUtility -
            optimisticSelfUtility -
            0.10
        );
        double visitConfidence =
          (double)normalEdgeVisits /
          (normalEdgeVisits + 8.0);
        bestProposal *= exp(
          -searchParams.multiHeadTacticalDisproofStrength *
          utilityGap *
          visitConfidence
        );
      }
      double desiredVisits =
        searchParams.multiHeadTacticalRootVisits * bestProposal;
      rootAuxDesiredVisits[movePos] =
        std::max(rootAuxDesiredVisits[movePos],desiredVisits);
    }
  }

  double bestNormalSelfUtility = node.nextPla == P_WHITE ? parentUtility : -parentUtility;
  if(isRoot && searchParams.multiHeadVctUseNormalRules) {
    bool foundNormalChild = false;
    for(int i = 0; i<childrenCapacity; i++) {
      const SearchNode* child = children[i].getIfAllocated();
      if(child == NULL)
        break;
      int64_t edgeVisits = children[i].getEdgeVisits();
      int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
      double childWeight = child->stats.getChildWeight(edgeVisits,childVisits);
      if(edgeVisits <= 0 || childVisits <= 0 || childWeight <= 0.0)
        continue;

      double childUtility = child->stats.utilityAvg.load(std::memory_order_acquire);
      double childNoResult = child->stats.noResultValueAvg.load(std::memory_order_acquire);
      double parentNoResult = node.stats.noResultValueAvg.load(std::memory_order_acquire);
      double noResultReduce = searchParams.noResultUtilityReduce * (1.0 - parentNoResult);
      childUtility -=
        childNoResult *
        noResultUtilityDecrease(
          searchParams.noResultUtilityForWhite,noResultReduce,node.nextPla
        );
      double childSelfUtility =
        node.nextPla == P_WHITE ? childUtility : -childUtility;
      if(!foundNormalChild || childSelfUtility > bestNormalSelfUtility)
        bestNormalSelfUtility = childSelfUtility;
      foundNormalChild = true;
    }
  }

  auto getSelectionValueForObjective = [&](float nnPolicyProb, const SearchNode* child, Loc moveLoc,
                                           double childEdgeVisits, double childWeight,
                                           double childUtilityAvgWhitePerspective, double exploreScalingForObjective,
                                           double totalChildWeightForObjective, double fpuValueForObjective,
                                           double parentWeightPerVisitForObjective, double maxChildWeightForObjective) {
    int32_t childVirtualLosses = child->virtualLosses.load(std::memory_order_acquire);
    int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);

    double childUtility;
    if(childVisits <= 0 || childWeight <= 0.0)
      childUtility = fpuValueForObjective;
    else
      childUtility = childUtilityAvgWhitePerspective;

    //Virtual losses to direct threads down different paths
    if(childVirtualLosses > 0) {
      double virtualLossWeight = childVirtualLosses * searchParams.numVirtualLossesPerThread;

      double utilityRadius = searchParams.multiValueHeadUtilityMix == 0.0 ? searchParams.winLossUtilityFactor : 1.0;
      double virtualLossUtility = (node.nextPla == P_WHITE ? -utilityRadius : utilityRadius);
      double virtualLossWeightFrac = (double)virtualLossWeight / (virtualLossWeight + std::max(0.25,childWeight));
      childUtility = childUtility + (virtualLossUtility - childUtility) * virtualLossWeightFrac;
      childWeight += virtualLossWeight;
    }

    if(isRoot) {
      double desiredAuxVisits =
        rootAuxDesiredVisits[getPos(moveLoc)];
      if(desiredAuxVisits > 0.0 && childEdgeVisits < desiredAuxVisits)
        return 1e20;
      if(searchParams.futileVisitsThreshold > 0) {
        double requiredWeight = searchParams.futileVisitsThreshold * maxChildWeightForObjective;
        double averageVisitsPerWeight = (childEdgeVisits + 1.0) / (childWeight + parentWeightPerVisitForObjective);
        double estimatedRequiredVisits = requiredWeight * averageVisitsPerWeight;
        if(childVisits + thread.upperBoundVisitsLeft < estimatedRequiredVisits)
          return FUTILE_VISITS_PRUNE_VALUE;
      }
      double desiredVisitsCoeff =
        searchParams.rootDesiredPerChildVisitsCoeff +
        searchParams.multiHeadDrawRootMinVisitsCoeff *
          multiHeadDrawIntervalSignal;
      if(desiredVisitsCoeff > 0.0) {
        if(nnPolicyProb > 0 && childWeight < sqrt(nnPolicyProb * totalChildWeightForObjective * desiredVisitsCoeff)) {
          return 1e20;
        }
      }
      if(rootHintLoc != Board::NULL_LOC && moveLoc == rootHintLoc) {
        double averageWeightPerVisit = (childWeight + parentWeightPerVisitForObjective) / (childVisits + 1.0);
        int hintChildrenCapacity;
        const SearchChildPointer* hintChildren = node.getChildren(hintChildrenCapacity);
        for(int i = 0; i<hintChildrenCapacity; i++) {
          const SearchNode* c = hintChildren[i].getIfAllocated();
          if(c == NULL)
            break;
          int64_t cEdgeVisits = hintChildren[i].getEdgeVisits();
          double cWeight = c->stats.getChildWeight(cEdgeVisits);
          if(childWeight + averageWeightPerVisit < cWeight * 0.8)
            return 1e20;
        }
      }

      if(searchParams.wideRootNoise > 0.0 && nnPolicyProb >= 0) {
        maybeApplyWideRootNoise(childUtility, nnPolicyProb, searchParams, &thread, node);
      }
    }

    double forcedReplyVisits = isRoot ?
      searchParams.multiHeadDrawForcedReplyRootVisits :
      searchParams.multiHeadDrawForcedReplyTreeVisits;
    if(forcedReplyVisits > 0.0) {
      const NNOutput* childNNOutput = child->getNNOutput();
      if(childNNOutput != NULL) {
        if(!childNNOutput->hasPolicyByHead())
          throw StringError("multi-head forced-reply verification requires a v112 model");
        double peakReplyPolicy = 0.0;
        for(int replyPos = 0; replyPos<policySize; replyPos++) {
          double replyPolicy =
            childNNOutput->getPolicyProbMaybeNoised(replyPos);
          if(replyPolicy > peakReplyPolicy)
            peakReplyPolicy = replyPolicy;
        }
        double replyPolicyThreshold =
          searchParams.multiHeadDrawForcedReplyPolicyThreshold;
        double concentration = std::clamp(
          (peakReplyPolicy - replyPolicyThreshold) /
            (1.0 - replyPolicyThreshold),
          0.0,1.0
        );
        concentration =
          concentration * concentration * (3.0 - 2.0 * concentration);

        Player childPla = child->nextPla;
        double childMustWinProb = childPla == P_WHITE ?
          childNNOutput->whiteWinProbByHead[3] :
          childNNOutput->whiteLossProbByHead[3];
        double childLossProbWhenDrawCountsAsWin = childPla == P_WHITE ?
          childNNOutput->whiteLossProbByHead[2] :
          childNNOutput->whiteWinProbByHead[2];
        double childNonLossProb =
          1.0 - childLossProbWhenDrawCountsAsWin;
        double childOutcomeInterval = std::clamp(
          childNonLossProb - childMustWinProb,
          0.0,1.0
        );
        double childDrawSignal = sqrt(
          std::clamp(
            (double)childNNOutput->whiteNoResultProb,
            0.0,1.0
          ) *
          childOutcomeInterval
        );
        double desiredForcedReplyWeight =
          1.0 +
          forcedReplyVisits *
            concentration *
            childDrawSignal *
            forcedReplyAuxPolicyScale[getPos(moveLoc)];
        if(childWeight < desiredForcedReplyWeight)
          return 1e20;
      }
    }

    return getExploreSelectionValue(exploreScalingForObjective,nnPolicyProb,childWeight,childUtility,node.nextPla);
  };

  auto getNewSelectionValueForObjective = [&](float nnPolicyProb, double exploreScalingForObjective,
                                              double fpuValueForObjective, double parentWeightPerVisitForObjective,
                                              double maxChildWeightForObjective) {
    double childWeight = 0;
    double childUtility = fpuValueForObjective;
    if(&node == rootNode) {
      if(searchParams.futileVisitsThreshold > 0) {
        double averageVisitsPerWeight = 1.0 / parentWeightPerVisitForObjective;
        double requiredWeight = searchParams.futileVisitsThreshold * maxChildWeightForObjective;
        double estimatedRequiredVisits = requiredWeight * averageVisitsPerWeight;
        if(thread.upperBoundVisitsLeft < estimatedRequiredVisits)
          return FUTILE_VISITS_PRUNE_VALUE;
      }
      if(searchParams.wideRootNoise > 0.0) {
        maybeApplyWideRootNoise(childUtility, nnPolicyProb, searchParams, &thread, node);
      }
    }
    return getExploreSelectionValue(exploreScalingForObjective,nnPolicyProb,childWeight,childUtility,node.nextPla);
  };

  auto combineMultiValueHeadSelectionValues = [&](double whiteWinSelectionValue, double blackWinSelectionValue) {
    if(thread.normalObjective == P_WHITE)
      return whiteWinSelectionValue;
    if(thread.normalObjective == P_BLACK)
      return blackWinSelectionValue;
    double sideToMoveWinWeight =
      getSideToMoveWinObjectiveWeight(node);
    double otherWinWeight = 1.0 - sideToMoveWinWeight;
    double sideToMoveWinSelectionValue = node.nextPla == P_WHITE ?
      whiteWinSelectionValue : blackWinSelectionValue;
    double otherWinSelectionValue = node.nextPla == P_WHITE ?
      blackWinSelectionValue : whiteWinSelectionValue;
    double sharpness = searchParams.multiHeadObjectiveSelectionSharpness;
    if(sharpness <= 0.0) {
      return
        sideToMoveWinSelectionValue * sideToMoveWinWeight +
        otherWinSelectionValue * otherWinWeight;
    }
    if(sideToMoveWinWeight <= 0.0)
      return otherWinSelectionValue;
    if(otherWinWeight <= 0.0)
      return sideToMoveWinSelectionValue;

    double maxSelectionValue = std::max(
      sideToMoveWinSelectionValue,otherWinSelectionValue
    );
    double expMean =
      sideToMoveWinWeight *
        exp(sharpness * (sideToMoveWinSelectionValue - maxSelectionValue)) +
      otherWinWeight *
        exp(sharpness * (otherWinSelectionValue - maxSelectionValue));
    return maxSelectionValue + log(expMean) / sharpness;
  };

  //Try all existing children
  //Also count how many children we actually find
  numChildrenFound = 0;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    numChildrenFound++;
    int64_t childEdgeVisits = children[i].getEdgeVisits();

    Loc moveLoc = children[i].getMoveLocRelaxed();
    bool isDuringSearch = true;
    double selectionValue;
    if(!useMultiValueHeads) {
      selectionValue = getExploreSelectionValueOfChild(
        node,
        getNormalSearchPolicyProb(getPos(moveLoc)),
        child,
        moveLoc,
        exploreScaling,
        totalChildWeight,childEdgeVisits,fpuValue,
        parentUtility,parentWeightPerVisit,
        isDuringSearch,maxChildWeight,&thread
      );
    }
    else {
      int movePos = getPos(moveLoc);
      float whiteWinPolicyProb = getWhiteWinSearchPolicyProb(movePos);
      float blackWinPolicyProb = getBlackWinSearchPolicyProb(movePos);
      int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
      int64_t rawWhiteWinEdgeVisits =
        children[i].getObjectiveEdgeVisits(P_WHITE);
      int64_t rawBlackWinEdgeVisits =
        children[i].getObjectiveEdgeVisits(P_BLACK);
      double crossWeight = searchParams.multiHeadObjectiveCrossWeight;
      double whiteWinEdgeVisits = separateObjectivePlayouts ?
        rawWhiteWinEdgeVisits + crossWeight * rawBlackWinEdgeVisits :
        (double)childEdgeVisits;
      double blackWinEdgeVisits = separateObjectivePlayouts ?
        rawBlackWinEdgeVisits + crossWeight * rawWhiteWinEdgeVisits :
        (double)childEdgeVisits;
      double whiteWinChildWeight = separateObjectivePlayouts ?
        (double)whiteWinEdgeVisits :
        child->stats.getChildWhiteWinWeight(childEdgeVisits,childVisits);
      double blackWinChildWeight = separateObjectivePlayouts ?
        (double)blackWinEdgeVisits :
        child->stats.getChildBlackWinWeight(childEdgeVisits,childVisits);
      double whiteWinUtility = child->stats.whiteWinUtilityAvg.load(std::memory_order_acquire);
      double blackWinUtilityInvAsWhite = child->stats.blackWinUtilityInvAvg.load(std::memory_order_acquire);
      double whiteWinSelectionValue = getSelectionValueForObjective(
        whiteWinPolicyProb, child, moveLoc, whiteWinEdgeVisits, whiteWinChildWeight, whiteWinUtility,
        whiteWinExploreScaling, totalWhiteWinChildWeight, whiteWinFpuValue,
        whiteWinParentWeightPerVisit, maxWhiteWinChildWeight
      );
      double blackWinSelectionValue = getSelectionValueForObjective(
        blackWinPolicyProb, child, moveLoc, blackWinEdgeVisits, blackWinChildWeight, blackWinUtilityInvAsWhite,
        blackWinExploreScaling, totalBlackWinChildWeight, blackWinFpuValue,
        blackWinParentWeightPerVisit, maxBlackWinChildWeight
      );
      selectionValue = combineMultiValueHeadSelectionValues(whiteWinSelectionValue, blackWinSelectionValue);
    }
    selectionValue += getAuxiliaryValueExploreBonus(
      child,getPos(moveLoc),childEdgeVisits
    );
    selectionValue += getGuidedValueSelectionAdjustment(child);
    selectionValue += getOutcomeIntervalExploreBonus(child,childEdgeVisits);
    double vctValidationPhaseScale = searchParams.multiHeadVctUseNormalRules ?
      getVctValidationPhaseScale(thread) :
      1.0;
    if(
      searchParams.multiHeadVctValidationProp > 0.0 &&
      vctValidationPhaseScale > 0.0
    ) {
      int64_t whiteVctVisits = children[i].getVctEdgeVisits(P_WHITE);
      int64_t blackVctVisits = children[i].getVctEdgeVisits(P_BLACK);
      auto getVctUtilityEvidence = [&](Player attacker, int64_t edgeVctVisits) {
        if(edgeVctVisits <= 0)
          return 0.0;
        VctStats stats(child->getVctStats(attacker));
        if(stats.visits <= 0 || stats.weightSum <= 0.0)
          return 0.0;
        double successProb = attacker == P_WHITE ?
          0.5 * (stats.utilityAvg + 1.0) :
          0.5 * (1.0 - stats.utilityAvg);
        successProb = std::clamp(successProb,0.0,1.0);
        double variance = std::max(0.0,stats.utilitySqAvg - stats.utilityAvg * stats.utilityAvg);
        double effectiveSamples = stats.weightSqSum > 0.0 ?
          stats.weightSum * stats.weightSum / stats.weightSqSum :
          (double)stats.visits;
        effectiveSamples = std::max(1.0,effectiveSamples);
        double empiricalStdErr = 0.5 * sqrt(variance / effectiveSamples);
        double explorationStdErr = 0.25 / sqrt(effectiveSamples);
        double uncertainty = std::max(empiricalStdErr,explorationStdErr);
        double lowerBound = std::clamp(
          successProb - searchParams.multiHeadVctUcbCoeff * uncertainty,
          0.0,1.0
        );
        return std::max(0.0,2.0 * lowerBound - 1.0);
      };

      Player ownAttacker = node.nextPla;
      Player opponentAttacker = getOpp(node.nextPla);
      int64_t ownVisits = ownAttacker == P_WHITE ? whiteVctVisits : blackVctVisits;
      int64_t opponentVisits = opponentAttacker == P_WHITE ? whiteVctVisits : blackVctVisits;
      double validationSignal;
      if(isRoot && searchParams.multiHeadVctUseNormalRules) {
        VctStats stats(child->getVctStats(ownAttacker));
        double evidenceSelfUtility = -1.0;
        if(ownVisits > 0 && stats.visits > 0 && stats.weightSum > 0.0) {
          double selfUtility =
            ownAttacker == P_WHITE ? stats.utilityAvg : -stats.utilityAvg;
          double variance = std::max(
            0.0,
            stats.utilitySqAvg - stats.utilityAvg * stats.utilityAvg
          );
          double effectiveSamples = stats.weightSqSum > 0.0 ?
            stats.weightSum * stats.weightSum / stats.weightSqSum :
            (double)stats.visits;
          effectiveSamples = std::max(1.0,effectiveSamples);
          double empiricalStdErr = sqrt(variance / effectiveSamples);
          double explorationStdErr = 0.5 / sqrt(effectiveSamples);
          double uncertainty = std::max(empiricalStdErr,explorationStdErr);
          evidenceSelfUtility =
            selfUtility +
            searchParams.multiHeadVctValidationOptimism * uncertainty;
        }
        double improvement = std::max(
          0.0,
          evidenceSelfUtility - bestNormalSelfUtility
        );
        validationSignal = improvement * sqrt((double)std::max<int64_t>(0,ownVisits));
      }
      else {
        double ownEvidence = getVctUtilityEvidence(ownAttacker,ownVisits);
        double opponentEvidence = getVctUtilityEvidence(opponentAttacker,opponentVisits);
        validationSignal =
          ownEvidence * sqrt((double)ownVisits) -
          opponentEvidence * sqrt((double)opponentVisits);
      }
      selectionValue +=
        searchParams.multiHeadVctValidationUtility *
        searchParams.multiHeadVctValidationProp *
        vctValidationPhaseScale *
        validationSignal / (1.0 + childEdgeVisits);
    }
    if(selectionValue > maxSelectionValue) {
      maxSelectionValue = selectionValue;
      bestChildIdx = i;
      bestChildMoveLoc = moveLoc;
    }

    posesWithChildBuf[getPos(moveLoc)] = true;
  }

  const std::vector<int>& avoidMoveUntilByLoc = thread.pla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;

  //Try the new child with the best policy value
  Loc bestNewMoveLoc = Board::NULL_LOC;
  float bestNewNNPolicyProb = -1.0f;
  double bestNewObjectiveSelectionValue = POLICY_ILLEGAL_SELECTION_VALUE;
  double bestNewAuxDesiredVisits = 0.0;
  for(int movePos = 0; movePos<policySize; movePos++) {
    bool alreadyTried = posesWithChildBuf[movePos];
    if(alreadyTried)
      continue;

    Loc moveLoc = NNPos::posToLoc(movePos,thread.board.x_size,thread.board.y_size,nnXLen,nnYLen);
    if(moveLoc == Board::NULL_LOC)
      continue;

    //Special logic for the root
    if(isRoot) {
      assert(thread.board.pos_hash == rootBoard.pos_hash);
      assert(thread.pla == rootPla);
      if(!isAllowedRootMove(moveLoc))
        continue;
    }
    if(avoidMoveUntilByLoc.size() > 0) {
      assert(avoidMoveUntilByLoc.size() >= Board::MAX_ARR_SIZE);
      int untilDepth = avoidMoveUntilByLoc[moveLoc];
      if(thread.history.moveHistory.size() - rootHistory.moveHistory.size() < untilDepth)
        continue;
    }

    //Quit immediately for illegal moves
    float nnPolicyProb = getNormalSearchPolicyProb(movePos);
    if(nnPolicyProb < 0)
      continue;

    if(
      isRoot &&
      rootAuxDesiredVisits[movePos] >= 1.0
    ) {
      if(rootAuxDesiredVisits[movePos] > bestNewAuxDesiredVisits) {
        bestNewAuxDesiredVisits = rootAuxDesiredVisits[movePos];
        bestNewNNPolicyProb = nnPolicyProb;
        bestNewMoveLoc = moveLoc;
      }
      continue;
    }
    if(bestNewAuxDesiredVisits > 0.0)
      continue;

    if(useMultiValueHeads && objectivePolicyMix > 0.0) {
      float whiteWinPolicyProb = getWhiteWinSearchPolicyProb(movePos);
      float blackWinPolicyProb = getBlackWinSearchPolicyProb(movePos);
      double whiteWinSelectionValue = getNewSelectionValueForObjective(
        whiteWinPolicyProb, whiteWinExploreScaling, whiteWinFpuValue,
        whiteWinParentWeightPerVisit, maxWhiteWinChildWeight
      );
      double blackWinSelectionValue = getNewSelectionValueForObjective(
        blackWinPolicyProb, blackWinExploreScaling, blackWinFpuValue,
        blackWinParentWeightPerVisit, maxBlackWinChildWeight
      );
      double objectiveSelectionValue = combineMultiValueHeadSelectionValues(
        whiteWinSelectionValue,blackWinSelectionValue
      );
      if(objectiveSelectionValue > bestNewObjectiveSelectionValue) {
        bestNewObjectiveSelectionValue = objectiveSelectionValue;
        bestNewNNPolicyProb = nnPolicyProb;
        bestNewMoveLoc = moveLoc;
      }
    }
    else if(nnPolicyProb > bestNewNNPolicyProb) {
      bestNewNNPolicyProb = nnPolicyProb;
      bestNewMoveLoc = moveLoc;
    }
  }
  if(bestNewMoveLoc != Board::NULL_LOC) {
    double selectionValue;
    if(bestNewAuxDesiredVisits > 0.0) {
      selectionValue = 1e20;
    }
    else if(useMultiValueHeads && objectivePolicyMix > 0.0) {
      selectionValue = bestNewObjectiveSelectionValue;
    }
    else if(!useMultiValueHeads) {
      selectionValue = getNewExploreSelectionValue(
        node,
        exploreScaling,
        bestNewNNPolicyProb,fpuValue,
        parentWeightPerVisit,
        maxChildWeight,&thread
      );
    }
    else {
      double whiteWinSelectionValue = getNewSelectionValueForObjective(
        bestNewNNPolicyProb, whiteWinExploreScaling, whiteWinFpuValue,
        whiteWinParentWeightPerVisit, maxWhiteWinChildWeight
      );
      double blackWinSelectionValue = getNewSelectionValueForObjective(
        bestNewNNPolicyProb, blackWinExploreScaling, blackWinFpuValue,
        blackWinParentWeightPerVisit, maxBlackWinChildWeight
      );
      selectionValue = combineMultiValueHeadSelectionValues(whiteWinSelectionValue, blackWinSelectionValue);
    }
    if(selectionValue > maxSelectionValue) {
      maxSelectionValue = selectionValue;
      bestChildIdx = numChildrenFound;
      bestChildMoveLoc = bestNewMoveLoc;
    }
  }
}
