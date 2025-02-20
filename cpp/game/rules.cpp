#include "../game/rules.h"

#include "../external/nlohmann_json/json.hpp"

#include <sstream>

using namespace std;
using json = nlohmann::json;

Rules::Rules() {
  //Defaults if not set - closest match to TT rules
  scoringRule = SCORING_0;
  drawJudgeRule = DRAWJUDGE_DRAW;
  maxmoves = 0;
  maxmovesNoCapture = 200;//100 turns no capture
}

Rules::Rules(
  int sRule,
  int dRule
)
  :scoringRule(sRule), drawJudgeRule(dRule) 
{}

Rules::~Rules() {
}

bool Rules::operator==(const Rules& other) const {
  return
    scoringRule == other.scoringRule &&
    drawJudgeRule == other.drawJudgeRule &&
    maxmoves==other.maxmoves&&
    maxmovesNoCapture==other.maxmovesNoCapture;
}

bool Rules::operator!=(const Rules& other) const {
  return !(*this == other);
}


Rules Rules::getTrompTaylorish() {
  Rules rules;
  rules.scoringRule = SCORING_0;
  return rules;
}



set<string> Rules::scoringRuleStrings() {
  return {
    "0",
    "1",
    "2",
    "3"
  };
}

set<string> Rules::drawJudgeRuleStrings() {
  return {"DRAW", "COUNT", "WEIGHT"};
}

int Rules::parseScoringRule(const string& s) {
  if(s == "0") return Rules::SCORING_0;
  else if(s == "1") return Rules::SCORING_1;
  else if(s == "2") return Rules::SCORING_2;
  else if(s == "3") return Rules::SCORING_3;
  else throw IOError("Rules::parseScoringRule: Invalid scoring rule: " + s);
}

int Rules::parseDrawJudgeRule(const string& s) {
  if(s == "DRAW")
    return Rules::DRAWJUDGE_DRAW;
  else if(s == "COUNT")
    return Rules::DRAWJUDGE_COUNT;
  else if(s == "WEIGHT")
    return Rules::DRAWJUDGE_WEIGHT;
  else
    throw IOError("Rules::parseScoringRule: Invalid scoring rule: " + s);
}

string Rules::writeScoringRule(int scoringRule) {
  if(scoringRule == Rules::SCORING_0) return string("0");
  else if(scoringRule == Rules::SCORING_1) return string("1");
  else if(scoringRule == Rules::SCORING_2) return string("2");
  else if(scoringRule == Rules::SCORING_3) return string("3");
  return string("UNKNOWN");
}
string Rules::writeDrawJudgeRule(int s) {
  if(s == Rules::DRAWJUDGE_DRAW)
    return string("DRAW");
  else if(s == Rules::DRAWJUDGE_COUNT)
    return string("COUNT");
  else if(s == Rules::DRAWJUDGE_WEIGHT)
    return string("WEIGHT");
  return string("UNKNOWN");
}

ostream& operator<<(ostream& out, const Rules& rules) {
  out << "score" << Rules::writeScoringRule(rules.scoringRule);
  out << "drawjudge" << Rules::writeDrawJudgeRule(rules.drawJudgeRule);
  out << "mm" << rules.maxmoves;
  out << "mc" << rules.maxmovesNoCapture;
  return out;
}


string Rules::toString() const {
  ostringstream out;
  out << (*this);
  return out.str();
}

string Rules::toJsonString() const {
  return toJson().dump();
}

//omitDefaults: Takes up a lot of string space to include stuff, so omit some less common things if matches tromp-taylor rules
//which is the default for parsing and if not otherwise specified
json Rules::toJson() const {
  json ret;
  ret["scoring"] = writeScoringRule(scoringRule);
  ret["drawjudge"] = writeDrawJudgeRule(drawJudgeRule);
  ret["mm"] = maxmoves;
  ret["mc"] = maxmovesNoCapture;
  return ret;
}


Rules Rules::updateRules(const string& k, const string& v, Rules oldRules) {
  Rules rules = oldRules;
  string key = Global::trim(k);
  string value = Global::trim(Global::toUpper(v));
  if(key == "score") rules.scoringRule = Rules::parseScoringRule(value);
  else if(key == "scoring") rules.scoringRule = Rules::parseScoringRule(value);
  else if(key == "drawjudge") rules.drawJudgeRule = Rules::parseDrawJudgeRule(value);
  else if(key == "mm" || key == "maxmoves" ) rules.maxmoves = Global::stringToInt(v);
  else if(key == "mc" || key == "maxmovesnocapture" ) rules.maxmovesNoCapture = Global::stringToInt(v);
  else throw IOError("Unknown rules option: " + key);
  return rules;
}

static Rules parseRulesHelper(const string& sOrig) {
  Rules rules;
  string lowercased = Global::trim(Global::toLower(sOrig));
  
  if(lowercased == "tromp-taylor" || lowercased == "tromp_taylor" || lowercased == "tromp taylor" || lowercased == "tromptaylor") {
    rules.scoringRule = Rules::SCORING_0;
  }
  else if(sOrig.length() > 0 && sOrig[0] == '{') {
    //Default if not specified
    rules = Rules::getTrompTaylorish();
    try {
      json input = json::parse(sOrig);
      string s;
      for(json::iterator iter = input.begin(); iter != input.end(); ++iter) {
        string key = iter.key();
        if(key == "score")
          rules.scoringRule = Rules::parseScoringRule(iter.value().get<string>());
        else if(key == "scoring")
          rules.scoringRule = Rules::parseScoringRule(iter.value().get<string>());
        else if(key == "drawjudge")
          rules.drawJudgeRule = Rules::parseDrawJudgeRule(iter.value().get<string>());
        else if(key == "mm" || key == "maxmoves")
          rules.maxmoves = iter.value().get<int>();
        else if(key == "mc" || key == "maxmovesnocapture")
          rules.maxmovesNoCapture = iter.value().get<int>();
        else
          throw IOError("Unknown rules option: " + key);
      }
    }
    catch(nlohmann::detail::exception&) {
      throw IOError("Could not parse rules: " + sOrig);
    }
  }

  // This is more of a legacy internal format, not recommended for users to provide
  else {
    throw IOError("Could not parse rules: " + sOrig);
  }

  return rules;
}

string Rules::toStringMaybeNice() const {
  if(*this == parseRulesHelper("TrompTaylor"))
    return "TrompTaylor";
  return toString();
}

Rules Rules::parseRules(const string& sOrig) {
  return parseRulesHelper(sOrig);
}


bool Rules::tryParseRules(const string& sOrig, Rules& buf) {
  Rules rules;
  try { rules = parseRulesHelper(sOrig); }
  catch(const StringError&) { return false; }
  buf = rules;
  return true;
}




const Hash128 Rules::ZOBRIST_SCORING_RULE_HASH[5] = {
  Hash128(0xcf88edc467a2211dULL, 0x0e326e3d299adca3ULL),
  Hash128(0xfc14dd6517a48b60ULL, 0x506b26c37b1ea6f2ULL),
  Hash128(0x1e193f506f8d19b2ULL, 0x2a0b378aca939942ULL),
  Hash128(0xbdf8802e2eacadf9ULL, 0x6348beb7087a97a4ULL),
  Hash128(0x2a5c2e5db567ee33ULL, 0x07410cf0ae7613b8ULL),
};
const Hash128 Rules::ZOBRIST_DRAWJUDGE_RULE_HASH[3] = {
  Hash128(0xce4045e964e21dd4ULL, 0x911c4ea3eb546d81ULL),
  Hash128(0x42538d2b7a724859ULL, 0xac9dce2669396872ULL),
  Hash128(0x257b357c21b7c14fULL, 0xdbccc53a2414774eULL),
};