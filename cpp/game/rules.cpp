#include "../game/rules.h"

#include "../external/nlohmann_json/json.hpp"

#include <cmath>
#include <sstream>

using namespace std;
using json = nlohmann::json;

Rules::Rules() {
  basicRule = BASICRULE_DEFAULT;
  multiStoneSuicideLegal = true;
  komi = 7.5f;
}

Rules::Rules(
  int bRule,
  bool suic,
  float km
)
  :basicRule(bRule),
   multiStoneSuicideLegal(suic),
   komi(km)
{}

Rules::~Rules() {
}

bool Rules::operator==(const Rules& other) const {
  return
    basicRule == other.basicRule &&
    multiStoneSuicideLegal == other.multiStoneSuicideLegal &&
    komi == other.komi;
}

bool Rules::operator!=(const Rules& other) const {
  return !(*this == other);
}

Rules Rules::getTrompTaylorish() {
  Rules rules;
  rules.basicRule = BASICRULE_DEFAULT;
  rules.multiStoneSuicideLegal = true;
  rules.komi = 7.5f;
  return rules;
}

set<string> Rules::basicRuleStrings() {
  return {"DEFAULT"};
}

int Rules::parseBasicRule(const string& s) {
  string value = Global::toUpper(s);
  if(value == "DEFAULT" || value == "BASICRULE_DEFAULT")
    return BASICRULE_DEFAULT;
  throw IOError("Rules::parseBasicRule: Invalid basic rule: " + s);
}

string Rules::writeBasicRule(int basicRule) {
  if(basicRule == BASICRULE_DEFAULT)
    return "DEFAULT";
  return "UNKNOWN";
}

bool Rules::komiIsIntOrHalfInt(float komi) {
  return std::isfinite(komi) && komi * 2 == (int)(komi * 2);
}

ostream& operator<<(ostream& out, const Rules& rules) {
  out << "basicrule" << Rules::writeBasicRule(rules.basicRule)
      << "sui" << rules.multiStoneSuicideLegal
      << "komi" << rules.komi;
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
  ret["basicrule"] = writeBasicRule(basicRule);
  ret["suicide"] = multiStoneSuicideLegal;
  ret["komi"] = komi;
  return ret;
}

Rules Rules::updateRules(const string& k, const string& v, Rules oldRules) {
  Rules rules = oldRules;
  string key = Global::toLower(Global::trim(k));
  string value = Global::trim(Global::toUpper(v));
  if(key == "basicrule" || key == "basicrules")
    rules.basicRule = Rules::parseBasicRule(value);
  else if(key == "suicide" || key == "multistonesuicidelegal")
    rules.multiStoneSuicideLegal = Global::stringToBool(value);
  else if(key == "komi") {
    float newKomi = oldRules.komi;
    bool suc = Global::tryStringToFloat(value, newKomi);
    if(suc && newKomi >= MIN_USER_KOMI && newKomi <= MAX_USER_KOMI && komiIsIntOrHalfInt(newKomi))
      rules.komi = newKomi;
    else
      throw IOError("Wrong komi: " + value + ", komi should be a half-integer in the supported range");
  }
  else
    throw IOError("Unknown rules option: " + key);
  return rules;
}

static Rules parseRulesHelper(const string& sOrig) {
  Rules rules = Rules::getTrompTaylorish();
  string lowercased = Global::trim(Global::toLower(sOrig));

  if(lowercased == "default" || lowercased == "tromp-taylor" || lowercased == "tromp_taylor" ||
     lowercased == "tromp taylor" || lowercased == "tromptaylor") {
    return rules;
  }
  else if(sOrig.length() > 0 && sOrig[0] == '{') {
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
  else {
    auto startsWithAndStrip = [](string& str, const string& prefix) {
      bool matches = str.length() >= prefix.length() && str.substr(0,prefix.length()) == prefix;
      if(matches)
        str = str.substr(prefix.length());
      str = Global::trim(str);
      return matches;
    };

    string s = Global::trim(sOrig);
    if(s.length() <= 0)
      throw IOError("Could not parse rules: " + sOrig);

    while(true) {
      if(s.length() <= 0)
        break;

      if(startsWithAndStrip(s,"basicrule")) {
        if(startsWithAndStrip(s,"default"))
          rules.basicRule = Rules::BASICRULE_DEFAULT;
        else
          throw IOError("Could not parse rules: " + sOrig);
        continue;
      }
      if(startsWithAndStrip(s,"komi")) {
        int endIdx = 0;
        while(endIdx < s.length() && !Global::isAlpha(s[endIdx]) && !Global::isWhitespace(s[endIdx]))
          endIdx++;
        float komi;
        bool suc = Global::tryStringToFloat(s.substr(0,endIdx), komi);
        if(!suc || !std::isfinite(komi) || komi > Rules::MAX_USER_KOMI || komi < Rules::MIN_USER_KOMI || !Rules::komiIsIntOrHalfInt(komi))
          throw IOError("Could not parse rules: " + sOrig);
        rules.komi = komi;
        s = s.substr(endIdx);
        s = Global::trim(s);
        continue;
      }
      if(startsWithAndStrip(s,"sui")) {
        if(startsWithAndStrip(s,"1"))
          rules.multiStoneSuicideLegal = true;
        else if(startsWithAndStrip(s,"0"))
          rules.multiStoneSuicideLegal = false;
        else
          throw IOError("Could not parse rules: " + sOrig);
        continue;
      }

      throw IOError("Could not parse rules: " + sOrig);
    }
  }

  return rules;
}

string Rules::toStringMaybeNice() const {
  if(*this == parseRulesHelper("tromp-taylor"))
    return "tromp-taylor";
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
};

const Hash128 Rules::ZOBRIST_MULTI_STONE_SUICIDE_HASH =
  Hash128(0xe95b8d8de573742fULL, 0xed23c0fc3c814fb4ULL);

const Hash128 Rules::ZOBRIST_KOMI_HASH_BASE =
  Hash128(0x8aba00580c378fe8ULL, 0x7f6c1210e74fb440ULL);
