#ifndef GAME_RULES_H_
#define GAME_RULES_H_

#include "../core/global.h"
#include "../core/hash.h"

#include "../external/nlohmann_json/json.hpp"

struct Rules {
  static const int PENTERULE_CLASSIC = 0; //classic Pente
  static const int PENTERULE_KERYO = 1;  // Keryo-Pente
  int penteRule;

  int blackTargetCap;  // if black capture these many white stones, black wins
  int whiteTargetCap;  // if white capture these many black stones, white wins

  static const int VCNRULE_NOVC = 0;
  static const int VCNRULE_VC1_B = 1;
  static const int VCNRULE_VC2_B = 2;
  static const int VCNRULE_VC3_B = 3;
  static const int VCNRULE_VC4_B = 4;
  static const int VCNRULE_VC1_W = 11;
  static const int VCNRULE_VC2_W = 12;
  static const int VCNRULE_VC3_W = 13;
  static const int VCNRULE_VC4_W = 14;
  int VCNRule;

  bool firstPassWin;  // 和棋的时候，先pass的获胜

  int maxMoves;  // 达到最大步数直接和棋，0不限制


  Rules();
  Rules(int penteRule, int blackTargetCap, int whiteTargetCap, int VCNRule, bool firstPassWin, int maxMoves);
  ~Rules();

  bool operator==(const Rules& other) const;
  bool operator!=(const Rules& other) const;

  static std::set<std::string> PenteRuleStrings();
  static std::set<std::string> VCNRuleStrings();

  static Rules getTrompTaylorish();

  static int parsePenteRule(std::string s);
  static std::string writePenteRule(int sixWinRule);

  static int parseVCNRule(std::string s);
  static std::string writeVCNRule(int VCNRule);


  static Rules parseRules(const std::string& str);
  static bool tryParseRules(const std::string& str, Rules& buf);

  static Rules updateRules(const std::string& key, const std::string& value, Rules priorRules);

  Color vcSide() const;
  int vcLevel() const;

  friend std::ostream& operator<<(std::ostream& out, const Rules& rules);
  std::string toString() const;
  std::string toStringMaybeNice() const;
  std::string toJsonString() const;
  nlohmann::json toJson() const;
  
  static const Hash128 ZOBRIST_PENTE_RULE_HASH[3];
  static const Hash128 ZOBRIST_BLACK_CAP_RULE_HASH_BASE;
  static const Hash128 ZOBRIST_WHITE_CAP_RULE_HASH_BASE;
  static const Hash128 ZOBRIST_VCNRULE_HASH_BASE;
  static const Hash128 ZOBRIST_FIRSTPASSWIN_HASH;
  static const Hash128 ZOBRIST_MAXMOVES_HASH_BASE;
};

#endif  // GAME_RULES_H_
