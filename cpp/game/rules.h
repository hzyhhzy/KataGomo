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

  int maxmoves;//draw if these many moves
  int maxmovesNoCapture;//draw if these many moves without capture



  Rules();
  Rules(
    int scoringRule
  );
  ~Rules();

  bool operator==(const Rules& other) const;
  bool operator!=(const Rules& other) const;

  bool equals(const Rules& other) const;
  bool gameResultWillBeInteger() const;

  static Rules getTrompTaylorish();
  static Rules getSimpleTerritory();

  static std::set<std::string> scoringRuleStrings();
  static int parseScoringRule(const std::string& s);
  static std::string writeScoringRule(int scoringRule);


  static Rules parseRules(const std::string& str);
  static bool tryParseRules(const std::string& str, Rules& buf);

  static Rules updateRules(const std::string& key, const std::string& value, Rules priorRules);

  friend std::ostream& operator<<(std::ostream& out, const Rules& rules);
  std::string toString() const;
  std::string toStringMaybeNice() const;
  std::string toJsonString() const;
  nlohmann::json toJson() const;

  static const Hash128 ZOBRIST_SCORING_RULE_HASH[2];

};

#endif  // GAME_RULES_H_
