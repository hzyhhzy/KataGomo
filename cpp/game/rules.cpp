#include "../game/rules.h"

#include "../external/nlohmann_json/json.hpp"

#include <sstream>

using namespace std;
using json = nlohmann::json;

Rules::Rules() {
  basicRule = BASICRULE_FREESTYLE;
  maxMoves = 0;
}

Rules::Rules(
  int basicRule,
  int maxMoves
)
  :basicRule(basicRule), maxMoves(maxMoves) {}

Rules::~Rules() {
}

bool Rules::operator==(const Rules& other) const {
  return basicRule == other.basicRule && maxMoves == other.maxMoves;
}

bool Rules::operator!=(const Rules& other) const {
  return !(*this == other);
}


Rules Rules::getTrompTaylorish() {
  Rules rules;
  return rules;
}

set<string> Rules::basicRuleStrings() {
  return {"FREESTYLE", "STANDARD", "CON7", "DCON5"};
}

int Rules::parseBasicRule(const string& s) {
  string value = Global::toUpper(s);
  if(value == "FREESTYLE")
    return BASICRULE_FREESTYLE;
  if(value == "STANDARD")
    return BASICRULE_STANDARD;
  if(value == "CON7")
    return BASICRULE_CON7;
  if(value == "DCON5")
    return BASICRULE_DCON5;
  throw IOError("Rules::parseBasicRule: Invalid basic rule: " + s);
}

string Rules::writeBasicRule(int basicRule) {
  if(basicRule == BASICRULE_FREESTYLE)
    return "FREESTYLE";
  if(basicRule == BASICRULE_STANDARD)
    return "STANDARD";
  if(basicRule == BASICRULE_CON7)
    return "CON7";
  if(basicRule == BASICRULE_DCON5)
    return "DCON5";
  return "UNKNOWN";
}
ostream& operator<<(ostream& out, const Rules& rules) {
  out << "basicrule" << Rules::writeBasicRule(rules.basicRule) << "maxmoves" << rules.maxMoves;
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
  ret["basicrule"] = writeBasicRule(basicRule);
  ret["maxmoves"] = maxMoves;
  return ret;
}


Rules Rules::updateRules(const string& k, const string& v, Rules oldRules) {
  Rules rules = oldRules;
  string key = Global::toLower(Global::trim(k));
  string value = Global::trim(Global::toUpper(v));
  if(key == "basicrule" || key == "basicrules")
    rules.basicRule = Rules::parseBasicRule(value);
  else if(key == "maxmoves") {
    int newMaxMoves = oldRules.maxMoves;
    bool suc = Global::tryStringToInt(value, newMaxMoves);
    if(suc && newMaxMoves >= 0)
      rules.maxMoves = newMaxMoves;
    else
      throw IOError("Wrong maxmoves: " + value + ", maxmoves should be a non-negative integer");
  } else
    throw IOError("Unknown rules option: " + key);
  return rules;
}

static Rules parseRulesHelper(const string& sOrig) {
  Rules rules;
  string lowercased = Global::trim(Global::toLower(sOrig));

  if(lowercased == "freestyle") {
    rules.basicRule = Rules::BASICRULE_FREESTYLE;
    rules.maxMoves = 0;
  }
  else if(lowercased == "standard") {
    rules.basicRule = Rules::BASICRULE_STANDARD;
    rules.maxMoves = 0;
  }
  else if(lowercased == "con7") {
    rules.basicRule = Rules::BASICRULE_CON7;
    rules.maxMoves = 0;
  }
  else if(lowercased == "dcon5") {
    rules.basicRule = Rules::BASICRULE_DCON5;
    rules.maxMoves = 0;
  }
  else if(sOrig.length() > 0 && sOrig[0] == '{') {
    rules = Rules::getTrompTaylorish();
    try {
      json input = json::parse(sOrig);
      for(json::iterator iter = input.begin(); iter != input.end(); ++iter) {
        string key = iter.key();
        string value = iter.value().is_string() ? iter.value().get<string>() : iter.value().dump();
        rules = Rules::updateRules(key, value, rules);
      }
    }
    catch(nlohmann::detail::exception&) {
      throw IOError("Could not parse rules: " + sOrig);
    }
  }

  //This is more of a legacy internal format, not recommended for users to provide
  else {
    auto startsWithAndStrip = [](string& str, const string& prefix) {
      bool matches = str.length() >= prefix.length() && str.substr(0,prefix.length()) == prefix;
      if(matches)
        str = str.substr(prefix.length());
      str = Global::trim(str);
      return matches;
    };

    //Default if not specified
    rules = Rules::getTrompTaylorish();

    string s = sOrig;
    s = Global::trim(s);

    //But don't allow the empty string
    if(s.length() <= 0)
      throw IOError("Could not parse rules: " + sOrig);

    while(true) {
      if(s.length() <= 0)
        break;

      if(startsWithAndStrip(s,"basicrule")) {
        auto ruleSet = Rules::basicRuleStrings();
        bool found = false;
        for(const string& rule : ruleSet) {
          if(startsWithAndStrip(s, Global::toLower(rule))) {
            rules.basicRule = Rules::parseBasicRule(rule);
            found = true;
            break;
          }
        }
        if(!found)
          throw IOError("Could not parse rules: " + sOrig);
        continue;
      }
      if(startsWithAndStrip(s, "maxmoves")) {
        int endIdx = 0;
        while(endIdx < s.length() && Global::isDigit(s[endIdx]))
          endIdx++;
        int maxMoves;
        bool suc = Global::tryStringToInt(s.substr(0, endIdx), maxMoves);
        if(!suc)
          throw IOError("Could not parse rules: " + sOrig);
        if(maxMoves < 0 || maxMoves > 100000000)
          throw IOError("Could not parse rules: " + sOrig);
        rules.maxMoves = maxMoves;
        s = s.substr(endIdx);
        s = Global::trim(s);
        continue;
      }

      //Unknown rules format
      else throw IOError("Could not parse rules: " + sOrig);
    }
  }

  return rules;
}

string Rules::toStringMaybeNice() const {
  if(*this == parseRulesHelper("freestyle"))
    return "freestyle";
  if(*this == parseRulesHelper("standard"))
    return "standard";
  if(*this == parseRulesHelper("con7"))
    return "con7";
  if(*this == parseRulesHelper("dcon5"))
    return "dcon5";
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




const Hash128 Rules::ZOBRIST_BASIC_RULE_HASH[Rules::NUM_BASIC_RULES] = {
  Hash128(0x72eeccc72c82a5e7ULL, 0x0d1265e413623e2bULL),
  Hash128(0x125bfe48a41042d5ULL, 0x061866b5f2b98a79ULL),
  Hash128(0xad2e6415c78086c7ULL, 0xe2d49ea6690a385cULL),
  Hash128(0xdcdaf38baaabd7d9ULL, 0x1197673d4b7593ffULL),
};
const Hash128 Rules::ZOBRIST_MAXMOVES_HASH_BASE =
  Hash128(0x8aba00580c378fe8ULL, 0x7f6c1210e74fb440ULL);
