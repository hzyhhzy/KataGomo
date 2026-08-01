#include "../game/rules.h"

#include "../external/nlohmann_json/json.hpp"

#include <sstream>

using namespace std;
using json = nlohmann::json;

Rules::Rules()
  : maxMoves(200),
    repetitionCount(2),
    noLegalMoveRule(NO_LEGAL_MOVE_COUNT),
    komi(0) {}

Rules::Rules(int maxMoves_, int repetitionCount_, int noLegalMoveRule_)
  : maxMoves(maxMoves_),
    repetitionCount(repetitionCount_),
    noLegalMoveRule(noLegalMoveRule_),
    komi(0) {
  if(maxMoves < 0)
    throw IOError("Rules: maxMoves must be nonnegative");
  if(repetitionCount != 2 && repetitionCount != 3)
    throw IOError("Rules: repetitionCount must be 2 or 3");
  if(noLegalMoveRule < NO_LEGAL_MOVE_LOSE || noLegalMoveRule > NO_LEGAL_MOVE_COUNT)
    throw IOError("Rules: invalid noLegalMoveRule");
}

Rules::~Rules() {}

bool Rules::operator==(const Rules& other) const {
  return
    maxMoves == other.maxMoves &&
    repetitionCount == other.repetitionCount &&
    noLegalMoveRule == other.noLegalMoveRule;
}

bool Rules::operator!=(const Rules& other) const {
  return !(*this == other);
}

Rules Rules::getTrompTaylorish() {
  return Rules();
}

map<string,int> Rules::noLegalMoveRuleStringsMap() {
  return {
    pair<string,int>("LOSE", NO_LEGAL_MOVE_LOSE),
    pair<string,int>("DRAW", NO_LEGAL_MOVE_DRAW),
    pair<string,int>("COUNT", NO_LEGAL_MOVE_COUNT),
  };
}

set<string> Rules::noLegalMoveRuleStrings() {
  set<string> result;
  for(const auto& entry: noLegalMoveRuleStringsMap())
    result.insert(entry.first);
  return result;
}

int Rules::parseNoLegalMoveRule(const string& sOrig) {
  string s = Global::toUpper(Global::trim(sOrig));
  auto ruleMap = noLegalMoveRuleStringsMap();
  auto iter = ruleMap.find(s);
  if(iter == ruleMap.end())
    throw IOError("Rules::parseNoLegalMoveRule: invalid rule: " + sOrig);
  return iter->second;
}

string Rules::writeNoLegalMoveRule(int rule) {
  for(const auto& entry: noLegalMoveRuleStringsMap()) {
    if(entry.second == rule)
      return entry.first;
  }
  return "UNKNOWN";
}

static int parseBoundedInt(const string& value, const string& name, int minValue, int maxValue) {
  int result;
  if(!Global::tryStringToInt(Global::trim(value), result) || result < minValue || result > maxValue)
    throw IOError("Rules: invalid " + name + ": " + value);
  return result;
}

Rules Rules::updateRules(const string& k, const string& v, Rules oldRules) {
  Rules rules = oldRules;
  string key = Global::toLower(Global::trim(k));
  if(key == "mm" || key == "maxmoves")
    rules.maxMoves = parseBoundedInt(v, "maxMoves", 0, 1000000);
  else if(key == "rep" || key == "repeat" || key == "repetition" || key == "repetitioncount") {
    rules.repetitionCount = parseBoundedInt(v, "repetitionCount", 2, 3);
    if(rules.repetitionCount != 2 && rules.repetitionCount != 3)
      throw IOError("Rules: repetitionCount must be 2 or 3");
  }
  else if(key == "nolegal" || key == "nolegalmove" || key == "nolegalmoverule")
    rules.noLegalMoveRule = Rules::parseNoLegalMoveRule(v);
  else
    throw IOError("Unknown rules option: " + k);
  return rules;
}

ostream& operator<<(ostream& out, const Rules& rules) {
  out << "maxmoves" << rules.maxMoves;
  out << "repetition" << rules.repetitionCount;
  out << "nolegal" << Rules::writeNoLegalMoveRule(rules.noLegalMoveRule);
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

json Rules::toJson() const {
  json ret;
  ret["maxmoves"] = maxMoves;
  ret["repetition"] = repetitionCount;
  ret["nolegal"] = writeNoLegalMoveRule(noLegalMoveRule);
  return ret;
}

static Rules parseRulesHelper(const string& sOrig) {
  string trimmed = Global::trim(sOrig);
  string lower = Global::toLower(trimmed);
  if(
    lower == "surakarta" ||
    lower == "default" ||
    lower == "chinese-kgs" ||
    lower == "tromp-taylor" || lower == "tromp_taylor" ||
    lower == "tromp taylor" || lower == "tromptaylor"
  )
    return Rules();

  if(trimmed.empty() || trimmed[0] != '{')
    throw IOError("Could not parse rules: " + sOrig);

  Rules rules;
  try {
    json input = json::parse(trimmed);
    if(!input.is_object())
      throw IOError("Could not parse rules: " + sOrig);
    for(json::iterator iter = input.begin(); iter != input.end(); ++iter) {
      string value;
      if(iter.value().is_string())
        value = iter.value().get<string>();
      else if(iter.value().is_number_integer())
        value = Global::int64ToString(iter.value().get<int64_t>());
      else
        throw IOError("Rules value must be a string or integer: " + iter.key());
      rules = Rules::updateRules(iter.key(), value, rules);
    }
  }
  catch(const IOError&) {
    throw;
  }
  catch(const nlohmann::detail::exception&) {
    throw IOError("Could not parse rules: " + sOrig);
  }
  return rules;
}

string Rules::toStringMaybeNice() const {
  if(*this == Rules())
    return "Surakarta";
  return toString();
}

Rules Rules::parseRules(const string& sOrig) {
  return parseRulesHelper(sOrig);
}

bool Rules::tryParseRules(const string& sOrig, Rules& buf) {
  try {
    buf = parseRulesHelper(sOrig);
    return true;
  }
  catch(const StringError&) {
    return false;
  }
}

const Hash128 Rules::ZOBRIST_NO_LEGAL_MOVE_RULE_HASH[3] = {
  Hash128(0xcfe353052ab23e7aULL, 0x243466cc5740fa07ULL),
  Hash128(0x3bdac963636f8efbULL, 0x7f5d9b5d76a70889ULL),
  Hash128(0xd4aecfb2904ed7d1ULL, 0x9adba41979253974ULL),
};

const Hash128 Rules::ZOBRIST_REPETITION_COUNT_HASH[4] = {
  Hash128(),
  Hash128(),
  Hash128(0x23af6fd73de24455ULL, 0x2339118e63d7a780ULL),
  Hash128(0x88a88d3f5d2b7b6bULL, 0x4b0123e673c52ad5ULL),
};
