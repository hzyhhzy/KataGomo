#ifndef GAME_RULES_H_
#define GAME_RULES_H_

#include "../core/global.h"
#include "../core/hash.h"

#include "../external/nlohmann_json/json.hpp"

struct Rules {
  static const int BASICRULE_FREESTYLE = 0;
  static const int BASICRULE_STANDARD = 1;
  int basicRule;

  int maxMoves;

  Rules();
  Rules(
    int basicRule,
    int maxMoves
  );
  ~Rules();

  bool operator==(const Rules& other) const;
  bool operator!=(const Rules& other) const;


  static Rules getTrompTaylorish();

  static std::set<std::string> basicRuleStrings();
  static int parseBasicRule(const std::string& s);
  static std::string writeBasicRule(int basicRule);


  static Rules parseRules(const std::string& str);
  static bool tryParseRules(const std::string& str, Rules& buf);

  static Rules updateRules(const std::string& key, const std::string& value, Rules priorRules);

  friend std::ostream& operator<<(std::ostream& out, const Rules& rules);
  std::string toString() const;
  std::string toStringMaybeNice() const;
  std::string toJsonString() const;
  nlohmann::json toJson() const;

  static const Hash128 ZOBRIST_BASIC_RULE_HASH[2];
  static const Hash128 ZOBRIST_MAXMOVES_HASH_BASE;

};

#endif  // GAME_RULES_H_
