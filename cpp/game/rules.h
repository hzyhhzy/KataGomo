#ifndef GAME_RULES_H_
#define GAME_RULES_H_

#include "../core/global.h"
#include "../core/hash.h"

#include "../external/nlohmann_json/json.hpp"

struct Rules {

  //rules variants
  //too lazy to give them names
  static const int SCORING_0 = 0; //default: rat on land and river CANNOT eat each other, lion and tiger CANNOT jump rat of same player
  static const int SCORING_1 = 1; //rat on land and river CAN eat each other, lion and tiger CANNOT jump rat of same player
  static const int SCORING_2 = 2; //rat on land and river CANNOT eat each other, lion and tiger CAN jump rat of same player
  static const int SCORING_3 = 3; //rat on land and river CAN eat each other, lion and tiger CAN jump rat of same player
  static const int SCORING_4 = 4; //reserve
  int scoringRule;

  static const int DRAWJUDGE_DRAW = 0;  // draw is draw:
  static const int DRAWJUDGE_COUNT = 1;  // when draw, who has more pieces wins:
  static const int DRAWJUDGE_WEIGHT = 2;  // when draw, the 8 type of pieces have different weights: 3 1 1 1 1 2 4 5:
  int drawJudgeRule;

  static const int LOOPRULE_SEVENTHREE = 0; //7-3违例
  static const int LOOPRULE_NONE = 1; //允许无限循环（不推荐）
  static const int LOOPRULE_REPEATEND = 2; //检查8回合内的重复，若某一步甲方走完之后（局面A）与先前某回合（局面B）一致，则检查B的前一步甲方走的棋子是否与A的前一步甲方的棋子相同。若相同则乙胜，否则甲胜。
  static const int LOOPRULE_TWOONE = 3; //7-3违例改成2-1违例
  static const int LOOPRULE_FIVETWO = 4; //5-2违例
  int loopRule;

  int maxmoves;//draw if these many moves
  int maxmovesNoCapture;//draw if these many moves without capture



  Rules();
  Rules(int scoringRule, int drawJudgeRule, int loopRule
  );
  ~Rules();

  bool operator==(const Rules& other) const;
  bool operator!=(const Rules& other) const;

  bool equals(const Rules& other) const;
  bool gameResultWillBeInteger() const;

  static Rules getTrompTaylorish();
  static Rules getSimpleTerritory();

  static std::set<std::string> scoringRuleStrings();
  static std::set<std::string> drawJudgeRuleStrings();
  static std::set<std::string> loopRuleStrings();


  static int parseScoringRule(const std::string& s);
  static std::string writeScoringRule(int scoringRule);
  static int parseDrawJudgeRule(const std::string& s);
  static std::string writeDrawJudgeRule(int s);
  static int parseLoopRule(const std::string& s);
  static std::string writeLoopRule(int s);


  static Rules parseRules(const std::string& str);
  static bool tryParseRules(const std::string& str, Rules& buf);

  static Rules updateRules(const std::string& key, const std::string& value, Rules priorRules);

  friend std::ostream& operator<<(std::ostream& out, const Rules& rules);
  std::string toString() const;
  std::string toStringMaybeNice() const;
  std::string toJsonString() const;
  nlohmann::json toJson() const;

  static const Hash128 ZOBRIST_SCORING_RULE_HASH[5];
  static const Hash128 ZOBRIST_DRAWJUDGE_RULE_HASH[3];
  static const Hash128 ZOBRIST_LOOP_RULE_HASH[5];
  

};

#endif  // GAME_RULES_H_
