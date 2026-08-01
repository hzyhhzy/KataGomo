#ifndef GAME_RULES_H_
#define GAME_RULES_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../external/nlohmann_json/json.hpp"

struct Rules {
  // What happens when the player to move has no legal non-pass move.
  static const int NO_LEGAL_MOVE_LOSE = 0;
  static const int NO_LEGAL_MOVE_DRAW = 1;
  static const int NO_LEGAL_MOVE_COUNT = 2;

  // Zero disables the absolute move limit. A move is one completed turn by
  // one player, rather than one source/destination selection stage.
  int maxMoves;

  // End the game when a full position, including the side to move, has
  // occurred this many times. Supported values are 2 and 3.
  int repetitionCount;

  int noLegalMoveRule;

  // Retained only for compatibility with generic GTP/SGF/distributed code.
  // Surakarta scoring does not use komi.
  int komi;

  Rules();
  Rules(int maxMoves, int repetitionCount, int noLegalMoveRule);
  ~Rules();

  bool operator==(const Rules& other) const;
  bool operator!=(const Rules& other) const;

  // Kept under this historical name because much of KataGo's generic
  // infrastructure asks for a default rules object through this method.
  static Rules getTrompTaylorish();

  static std::map<std::string, int> noLegalMoveRuleStringsMap();
  static std::set<std::string> noLegalMoveRuleStrings();
  static int parseNoLegalMoveRule(const std::string& s);
  static std::string writeNoLegalMoveRule(int rule);

  static Rules parseRules(const std::string& str);
  static bool tryParseRules(const std::string& str, Rules& buf);
  static Rules updateRules(const std::string& key, const std::string& value, Rules priorRules);

  friend std::ostream& operator<<(std::ostream& out, const Rules& rules);
  std::string toString() const;
  std::string toStringMaybeNice() const;
  std::string toJsonString() const;
  nlohmann::json toJson() const;

  static const Hash128 ZOBRIST_NO_LEGAL_MOVE_RULE_HASH[3];
  static const Hash128 ZOBRIST_REPETITION_COUNT_HASH[4];
};

#endif  // GAME_RULES_H_
