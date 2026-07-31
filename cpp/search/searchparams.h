#ifndef SEARCH_SEARCHPARAMS_H_
#define SEARCH_SEARCHPARAMS_H_

#include "../core/global.h"
#include "../game/board.h"

struct SearchParams {
  //Utility function parameters
  double winLossUtilityFactor;     //Scaling for [-1,1] value for winning/losing
  double noResultUtilityForWhite; //Utility of having a no-result game (simple ko rules or nonterminating territory encore) 
  double multiValueHeadUtilityMix; //0 = legacy head0 utility, 1 = use v112 multi-head win/loss utility
  double multiValueHeadSelectionBias; //-1 = opponent win objective, 0 = average, 1 = side-to-move win objective
  double multiHeadNormalPolicyHead1Mix; //Blend the independently distilled ordinary head1 policy into head0
  double multiHeadObjectiveSearchStrength; //0 = fixed two-value mixture, 1 = dynamically choose among win, non-loss, and VCT objectives
  double multiHeadObjectiveSelectionPower; //Sharpness of the soft choice between normal win and non-loss objectives
  double multiHeadObjectiveSelectionSharpness; //Risk-seeking log-sum-exp sharpness across the weighted win and non-loss PUCT values
  double multiHeadObjectivePolicyMix; //Head3 policy for win and head2 policy for non-loss, mixed independently into their PUCT objectives
  bool multiHeadObjectiveSeparatePlayouts; //Give each normal playout to one win objective and aggregate it with objective-specific edge visits
  double multiHeadObjectiveCrossWeight; //Credit a separated playout to the other normal objective before objective-specific value downweighting
  double multiHeadObjectiveValueWeightExponent; //Bad-child downweighting for the independently aggregated normal objectives
  double multiHeadObjectiveVctWeight; //Relative worth of each VCT objective when dividing playouts among the four objectives
  double multiHeadVctObjectiveBudgetMix; //0 = confidence-only VCT budget, 1 = blend toward four-objective worth according to objective search strength
  double multiHeadVctMaxAttackProp; //Maximum fraction of playouts for side-to-move VCT search
  double multiHeadVctMaxDefenseProp; //Maximum fraction of playouts for opponent VCT search
  double multiHeadVctUcbCoeff; //Uncertainty bonus when deciding the VCT playout fraction
  double multiHeadVctProbScale; //VCT probability scale for the smooth budget curve
  double multiHeadVctProbPower; //Sharpness of the smooth VCT budget curve
  double multiHeadVctMainWinSuppression; //Suppress redundant VCT probing when head0 already expects the attacker to win
  double multiHeadVctPolicyMix; //0 = head0 policy, 1 = VCT policy head
  double multiHeadVctPolicyConsensusMix; //Blend VCT policy with its matching win/non-loss policy using normalized geometric consensus
  bool multiHeadVctUseNormalRules; //Use normal rules and head0 value for isolated VCT candidate validation
  double multiHeadVctProbeStartFraction; //Search fraction at which isolated VCT probing begins
  double multiHeadVctProbeEndFraction; //Search fraction at which isolated VCT probing ends
  double multiHeadVctProbeRampFraction; //Soft-ramp width at each VCT probe phase boundary
  double multiHeadVctPriorVisits; //Effective samples over which the VCT prior remains active
  double multiHeadVctNormalPolicyMaxMix; //Maximum VCT policy mixture used directly by normal search
  double multiHeadVctAttackPolicyScale; //Scale for head4 own-attack policy proposals
  double multiHeadVctDefensePolicyScale; //Scale for head5 opponent-attack defense policy proposals
  double multiHeadVctPolicyRawProbMix; //0 = shaped VCT confidence, 1 = raw VCT win probability
  double multiHeadVctNormalPolicyPow; //Softness of the VCT-probability policy mixture
  double multiHeadDrawWinNormalPolicyMaxMix; //Maximum head2 (draw counts as win) policy mixture in normal search
  double multiHeadDrawLossNormalPolicyMaxMix; //Maximum head3 (draw counts as loss) policy mixture in normal search
  double multiHeadDrawNormalPolicyProbScale; //Probability scale for head2/head3 normal policy mixtures
  double multiHeadDrawPolicyRawProbMix; //0 = shaped draw-head confidence, 1 = raw objective probability
  double multiHeadDrawNormalPolicyPow; //Softness of head2/head3 normal policy mixtures
  double multiHeadDrawPolicyFlattening; //Maximum head0 policy flattening in positions where head2/head3 expose a wide draw interval
  double multiHeadDrawRootMinVisitsCoeff; //Additional root child minimum-visit funnel, softly gated by the head2/head3 draw interval
  double multiHeadDrawAuxRootVisits; //Bounded root seed visits proposed by dynamic h2/h5 and h3/h4 policy consensus
  double multiHeadTacticalRootVisits; //Root seed visits proposed by the union of win, non-loss, and VCT auxiliary objectives
  double multiHeadTacticalContrastiveMix; //Prefer moves promoted by an auxiliary rule head relative to the normal h0/h1 ensemble
  int multiHeadTacticalObjectiveMask; //Bits 0..3 enable tactical root proposals from heads 2..5
  double multiHeadTacticalDisproofStrength; //Softly stop tactical root verification once normal-rule evidence disproves the candidate
  double multiHeadTacticalPolicyPower; //Concentrate tactical verification on the auxiliary policy peak
  double multiHeadDrawForcedReplyRootVisits; //Extra root visits for drawish children whose opponent reply policy is highly concentrated
  double multiHeadDrawForcedReplyTreeVisits; //Extra non-root visits that continue verifying concentrated forced-reply sequences
  double multiHeadDrawForcedReplyAuxPolicyGate; //Softly focus root forced-reply visits using head3/head4 attack and head2/head5 defense consensus
  double multiHeadDrawForcedReplySidecarVisits; //Isolated normal-rules visits for each root forced-reply candidate
  double multiHeadDrawForcedReplySidecarMaxProp; //Maximum total search fraction spent on isolated forced-reply candidates
  double multiHeadDrawForcedReplyPolicyThreshold; //Reply-policy peak where forced-reply verification begins
  double multiHeadVctGuidedPlayoutProp; //Maximum softly gated fraction of fixed-attacker head4/head5 playouts
  double multiHeadDrawGuidedPlayoutProp; //Maximum softly gated fraction of fixed-winner head3/head2 playouts
  double multiHeadGuidedStartFraction; //Search fraction before guided playout probability begins ramping up
  double multiHeadGuidedPolicyMix; //Policy mixture within fixed-player guided playouts
  double multiHeadGuidedValueWeight; //Auxiliary-vs-head0 value residual used within guided playout selection
  double multiHeadOutcomeIntervalExplore; //Decaying exploration bonus from head2/head3 decisive-outcome bounds
  double multiHeadOutcomeIntervalVisitScale; //Edge visits over which outcome-interval exploration decays
  double multiHeadAuxPolicyChildGate; //How strongly child NN values retire disproven auxiliary policy hints
  double multiHeadAuxPolicyVisitScale; //Number of edge visits over which auxiliary policy hints progressively decay
  double multiHeadAuxPolicyOptimism; //Strength of pointwise optimistic policy proposals from auxiliary heads
  double multiHeadAuxPolicyConcentrationScale; //Policy peak excess needed for an auxiliary head to be trusted
  double multiHeadAuxValueExplore; //Decaying selection bonus when an auxiliary value confirms its policy proposal
  double multiHeadAuxValueVisitScale; //Edge visits over which the auxiliary value bonus decays
  double multiHeadVctCpuctScale; //Exploration scaling within the isolated VCT search planes
  double multiHeadVctValueWeightExponent; //Bad-child downweighting within VCT search planes
  double multiHeadVctValidationProp; //Strength of the VCT-to-normal validation signal
  double multiHeadVctValidationUtility; //Maximum utility-scale validation bonus
  double multiHeadVctValidationOptimism; //Uncertainty multiples added to isolated evidence before normal validation
  double multiHeadVctNormalVerificationProp; //Maximum fraction of playouts that normally verify the best isolated root VCT candidate
  double multiHeadVctMoveSelectionWeight; //Root move-selection weight for high-confidence VCT evidence
  double multiHeadVctRelativeMoveSelectionWeight; //Root move-selection weight when a normal-value sidecar beats the main root estimate
  double multiHeadVctNnMoveSelectionWeight; //Root move-selection weight for root/child NN VCT confirmation
  double multiHeadVctMoveSelectionVisitScale; //VCT visits needed for root move-selection evidence
  
  double noResultUtilityReduce;  // Decrease draw utility for both side (if positive)

  //Search tree exploration parameters
  double cpuctExploration;  //Constant factor on exploration, should also scale up linearly with magnitude of utility
  double cpuctExplorationLog; //Constant factor on log-scaling exploration, should also scale up linearly with magnitude of utility
  double cpuctExplorationBase; //Scale of number of visits at which log behavior starts having an effect

  double cpuctUtilityStdevPrior;
  double cpuctUtilityStdevPriorWeight;
  double cpuctUtilityStdevScale;

  double fpuReductionMax;   //Max amount to reduce fpu value for unexplore children
  double fpuLossProp; //Scale fpu this proportion of the way towards assuming a move is a loss.

  bool fpuParentWeightByVisitedPolicy; //For fpu, blend between parent average and parent nn value based on proportion of policy visited.
  double fpuParentWeightByVisitedPolicyPow; //If fpuParentWeightByVisitedPolicy, what power to raise the proportion of policy visited for blending.
  double fpuParentWeight; //For fpu, 0 = use parent average, 1 = use parent nn value, interpolates between.

  //Tree value aggregation parameters
  double valueWeightExponent; //Amount to apply a downweighting of children with very bad values relative to good ones
  bool useNoisePruning; //For computation of value, prune out weight that greatly exceeds what is justified by policy prior
  double noisePruneUtilityScale; //The scale of the utility difference at which useNoisePruning has effect
  double noisePruningCap; //Maximum amount of weight that noisePruning can remove

  //Uncertainty weighting
  bool useUncertainty; //Weight visits by uncertainty
  double uncertaintyCoeff; //The amount of visits weight that an uncertainty of 1 utility is.
  double uncertaintyExponent; //Visits weight scales inversely with this power of the uncertainty
  double uncertaintyMaxWeight; //Add minimum uncertainty so that the most weight a node can have is this

  //Graph search
  bool useGraphSearch; //Enable graph search instead of tree search?
  double graphSearchCatchUpLeakProb; //Chance to perform a visit to deepen a branch anyways despite being behind on visit count.
  //double graphSearchCatchUpProp; //When sufficiently far behind on visits on a transposition, catch up extra by adding up to this fraction of parents visits at once.

  //Root parameters
  bool rootNoiseEnabled;
  double rootDirichletNoiseTotalConcentration; //Same as alpha * board size, to match alphazero this might be 0.03 * 361, total number of balls in the urn
  double rootDirichletNoiseWeight; //Policy at root is this weight * noise + (1 - this weight) * nn policy

  double rootPolicyTemperature; //At the root node, scale policy probs by this power
  double rootPolicyTemperatureEarly; //At the root node, scale policy probs by this power, early in the game
  double rootFpuReductionMax; //Same as fpuReductionMax, but at root
  double rootFpuLossProp; //Same as fpuLossProp, but at root
  int rootNumSymmetriesToSample; //For the root node, sample this many random symmetries (WITHOUT replacement) and average the results together.
  bool rootSymmetryPruning; //For the root node, search only one copy of each symmetrically equivalent move.
  //We use the min of these two together, and also excess visits get pruned if the value turns out bad.
  double rootDesiredPerChildVisitsCoeff; //Funnel sqrt(this * policy prob * total visits) down any given child that receives any visits at all at the root

  //Parameters for choosing the move to play
  double chosenMoveTemperature; //Make move roughly proportional to visit count ** (1/chosenMoveTemperature)
  double chosenMoveTemperatureEarly; //Temperature at start of game
  double chosenMoveTemperatureHalflife; //Halflife of decay from early temperature to temperature for the rest of the game, scales for board sizes other than 19.
  double chosenMoveSubtract; //Try to subtract this many visits from every move prior to applying temperature
  double chosenMovePrune; //Outright prune moves that have fewer than this many visits

  bool useLcbForSelection; //Using LCB for move selection?
  double lcbStdevs; //How many stdevs a move needs to be better than another for LCB selection
  double minVisitPropForLCB; //Only use LCB override when a move has this proportion of visits as the top move
  bool useNonBuggyLcb; //LCB was very minorly buggy as of pre-v1.8. Set to true to fix.

  //Mild behavior hackery
  bool rootPruneUselessMoves; //Prune moves that are entirely useless moves that prolong the game.
  double wideRootNoise; //Explore at the root more widely

  double playoutDoublingAdvantage; //Play as if we have this many doublings of playouts vs the opponent
  Player playoutDoublingAdvantagePla; //Negate playoutDoublingAdvantage when making a move for the opponent of this player. If empty, opponent of the root player.

  double avoidRepeatedPatternUtility; //Have the root player avoid repeating similar shapes, penalizing this much utility per instance.

  float nnPolicyTemperature; //Scale neural net policy probabilities by this temperature, applies everywhere in the tree

  bool useVCFInput;        // whether calculate VCF
  bool useForbiddenInput;  // whether use forbiddenPoints feature
  bool useHistoryInput;  // whether use forbiddenPoints feature
  double fourAttackPolicyReduce; //reduce policy of four attack, *exp(-x)


  double subtreeValueBiasFactor; //Dynamically adjust neural net utilties based on empirical stats about their errors in search
  int32_t subtreeValueBiasTableNumShards; //Number of shards for subtreeValueBiasFactor for initial hash lookup and mutexing
  double subtreeValueBiasFreeProp; //When a node is no longer part of the relevant search tree, only decay this proportion of the weight.
  double subtreeValueBiasWeightExponent; //When computing empiricial bias, weight subtree results by childvisits to this power.

  //Threading-related
  int nodeTableShardsPowerOfTwo; //Controls number of shards of node table for graph search transposition lookup
  double numVirtualLossesPerThread; //Number of virtual losses for one thread to add

  //Asyncbot
  int numThreads; //Number of threads
  int backendNumThreads;  // Number of threads for the backend, only used for ONNX-cpu backend
  int64_t maxVisits; //Max number of playouts from the root to think for, counting earlier playouts from tree reuse
  int64_t maxPlayouts; //Max number of playouts from the root to think for, not counting earlier playouts from tree reuse
  double maxTime; //Max number of seconds to think for

  //Same caps but when pondering
  int64_t maxVisitsPondering;
  int64_t maxPlayoutsPondering;
  double maxTimePondering;

  //Amount of time to reserve for lag when using a time control
  double lagBuffer;

  //Time control
  double treeReuseCarryOverTimeFactor; //Assume we gain this much "time" on the next move purely from % tree preserved * time spend on that tree.
  double overallocateTimeFactor; //Prefer to think this factor longer than recommended by base level time control
  double midgameTimeFactor; //Think this factor longer in the midgame, proportional to midgame weight
  double midgameTurnPeakTime; //The turn considered to have midgame weight 1.0, rising up from 0.0 in the opening, for 19x19
  double endgameTurnTimeDecay; //The scale of exponential decay of midgame weight back to 1.0, for 19x19
  double obviousMovesTimeFactor; //Think up to this factor longer on obvious moves, weighted by obviousness
  double obviousMovesPolicyEntropyTolerance; //What entropy does the policy need to be at most to be (1/e) obvious?
  double obviousMovesPolicySurpriseTolerance; //What logits of surprise does the search result need to be at most to be (1/e) obvious?

  double futileVisitsThreshold; //If a move would not be able to match this proportion of the max visits move in the time or visit or playout cap remaining, prune it.
  int64_t finishGameSearchDelayMicroseconds; //Avoid running "too fast" at the end of the game, to cost less CPU

  SearchParams();
  ~SearchParams();

  void printParams(std::ostream& out);

  //Params to use for testing, with some more recent values representative of more real use (as of Jan 2019)
  static SearchParams forTestsV1();
  //Params to use for testing, with some more recent values representative of more real use (as of Mar 2022)
  static SearchParams forTestsV2();

  static void failIfParamsDifferOnUnchangeableParameter(const SearchParams& initial, const SearchParams& dynamic);
};

#endif  // SEARCH_SEARCHPARAMS_H_
