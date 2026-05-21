#ifndef GAME_RULES_H_
#define GAME_RULES_H_

#include "../core/global.h"
#include "../core/hash.h"

#include "../external/nlohmann_json/json.hpp"

struct Rules {
  static const int BASICRULE_DEFAULT = 0;
  static const int NUM_BASIC_RULES = 1;
  int basicRule;

  bool multiStoneSuicideLegal;
  float komi;

  static constexpr float MIN_USER_KOMI = -750.0f;
  static constexpr float MAX_USER_KOMI = 750.0f;

  Rules();
  Rules(
    int basicRule,
    bool multiStoneSuicideLegal,
    float komi
  );
  ~Rules();

  bool operator==(const Rules& other) const;
  bool operator!=(const Rules& other) const;


  static Rules getTrompTaylorish();

  static std::set<std::string> basicRuleStrings();
  static int parseBasicRule(const std::string& s);
  static std::string writeBasicRule(int basicRule);
  static bool komiIsIntOrHalfInt(float komi);


  static Rules parseRules(const std::string& str);
  static bool tryParseRules(const std::string& str, Rules& buf);

  static Rules updateRules(const std::string& key, const std::string& value, Rules priorRules);

  friend std::ostream& operator<<(std::ostream& out, const Rules& rules);
  std::string toString() const;
  std::string toStringMaybeNice() const;
  std::string toJsonString() const;
  nlohmann::json toJson() const;

  static const Hash128 ZOBRIST_BASIC_RULE_HASH[NUM_BASIC_RULES];
  static const Hash128 ZOBRIST_MULTI_STONE_SUICIDE_HASH;
  static const Hash128 ZOBRIST_KOMI_HASH_BASE;

};

#endif  // GAME_RULES_H_
