#include "../game/rules.h"

#include "../external/nlohmann_json/json.hpp"

#include <sstream>

using namespace std;
using json = nlohmann::json;

Rules::Rules() {
  basicRule = BASICRULE_FREESTYLE;
  VCNRule = VCNRULE_NOVC;
  firstPassWin = false;
  maxMoves = 0;
}

Rules::Rules(int basicRule, int VCNRule, bool firstPassWin, int maxMoves)
  : basicRule(basicRule), VCNRule(VCNRule), firstPassWin(firstPassWin), maxMoves(maxMoves) {}

Rules::~Rules() {}

bool Rules::operator==(const Rules& other) const {
  return basicRule == other.basicRule && VCNRule == other.VCNRule && firstPassWin == other.firstPassWin &&
         maxMoves == other.maxMoves;
}

bool Rules::operator!=(const Rules& other) const {
  return !(*this == other);
}

Rules Rules::getTrompTaylorish() {
  Rules rules = Rules();
  return rules;
}

set<string> Rules::basicRuleStrings() {
  return {"FREESTYLE", "STANDARD", "RENJU"};
}
set<string> Rules::VCNRuleStrings() {
  return {"NOVC", "VC1B", "VC2B", "VC3B", "VC4B", "VCTB", "VCFB", "VC1W", "VC2W", "VC3W", "VC4W", "VCTW", "VCFW"};
}

int Rules::parseBasicRule(string s) {
  s = Global::toUpper(s);
  if(s == "FREESTYLE")
    return Rules::BASICRULE_FREESTYLE;
  else if(s == "STANDARD")
    return Rules::BASICRULE_STANDARD;
  else if(s == "RENJU")
    return Rules::BASICRULE_RENJU;
  else
    throw IOError("Rules::parseBasicRule: Invalid basic rule: " + s);
}

string Rules::writeBasicRule(int basicRule) {
  if(basicRule == Rules::BASICRULE_FREESTYLE)
    return string("FREESTYLE");
  if(basicRule == Rules::BASICRULE_STANDARD)
    return string("STANDARD");
  if(basicRule == Rules::BASICRULE_RENJU)
    return string("RENJU");
  return string("UNKNOWN");
}

int Rules::parseVCNRule(string s) {
  s = Global::toLower(s);
  if(s == "novc")
    return Rules::VCNRULE_NOVC;
  else if(s == "vc1b")
    return Rules::VCNRULE_VC1_B;
  else if(s == "vc2b")
    return Rules::VCNRULE_VC2_B;
  else if(s == "vc3b")
    return Rules::VCNRULE_VC3_B;
  else if(s == "vctb")
    return Rules::VCNRULE_VC3_B;
  else if(s == "vc4b")
    return Rules::VCNRULE_VC4_B;
  else if(s == "vcfb")
    return Rules::VCNRULE_VC4_B;
  else if(s == "vc1w")
    return Rules::VCNRULE_VC1_W;
  else if(s == "vc2w")
    return Rules::VCNRULE_VC2_W;
  else if(s == "vc3w")
    return Rules::VCNRULE_VC3_W;
  else if(s == "vctw")
    return Rules::VCNRULE_VC3_W;
  else if(s == "vc4w")
    return Rules::VCNRULE_VC4_W;
  else if(s == "vcfw")
    return Rules::VCNRULE_VC4_W;
  else
    throw IOError("Rules::parseVCNRule: Invalid VCN rule: " + s);
}

string Rules::writeVCNRule(int VCNRule) {
  if(VCNRule == Rules::VCNRULE_NOVC)
    return string("NOVC");
  if(VCNRule == Rules::VCNRULE_VC1_B)
    return string("VC1B");
  if(VCNRule == Rules::VCNRULE_VC2_B)
    return string("VC2B");
  if(VCNRule == Rules::VCNRULE_VC3_B)
    return string("VCTB");
  if(VCNRule == Rules::VCNRULE_VC4_B)
    return string("VCFB");
  if(VCNRule == Rules::VCNRULE_VC1_W)
    return string("VC1W");
  if(VCNRule == Rules::VCNRULE_VC2_W)
    return string("VC2W");
  if(VCNRule == Rules::VCNRULE_VC3_W)
    return string("VCTW");
  if(VCNRule == Rules::VCNRULE_VC4_W)
    return string("VCFW");
  return string("UNKNOWN");
}

Color Rules::vcSide() const {
  static_assert(VCNRULE_VC1_W == VCNRULE_VC1_B + 10, "Ensure VCNRule%10==N, VCNRule/10+1==color");
  if(VCNRule == VCNRULE_NOVC)
    return C_EMPTY;
  return 1 + VCNRule / 10;
}

int Rules::vcLevel() const {
  static_assert(VCNRULE_VC1_W == VCNRULE_VC1_B + 10, "Ensure VCNRule%10==N, VCNRule/10+1==color");
  if(VCNRule == VCNRULE_NOVC)
    return -1;
  return VCNRule % 10;
}

ostream& operator<<(ostream& out, const Rules& rules) {
  out << "basicrule" << Rules::writeBasicRule(rules.basicRule);
  out << "vcnrule" << Rules::writeVCNRule(rules.VCNRule);
  out << "firstpasswin" << rules.firstPassWin;
  out << "maxmoves" << rules.maxMoves;
  return out;
}

string Rules::toString() const {
  ostringstream out;
  out << (*this);
  return out.str();
}


json Rules::toJson() const {
  json ret;
  ret["basicrule"] = writeBasicRule(basicRule);
  ret["vcnrule"] = writeVCNRule(VCNRule);
  ret["firstpasswin"] = firstPassWin;
  ret["maxmoves"] = maxMoves;
  return ret;
}


string Rules::toJsonString() const {
  return toJson().dump();
}



Rules Rules::updateRules(const string& k, const string& v, Rules oldRules) {
  Rules rules = oldRules;
  string key = Global::toLower(Global::trim(k));
  string value = Global::trim(Global::toUpper(v));
  if(key == "basicrule")
    rules.basicRule = Rules::parseBasicRule(value);
  else if(key == "vcnrule") {
    rules.firstPassWin = false;
    rules.maxMoves = 0;
    rules.VCNRule = Rules::parseVCNRule(value);
  } else if(key == "firstpasswin") {
    rules.VCNRule = VCNRULE_NOVC;
    rules.maxMoves = 0;
    rules.firstPassWin = Global::stringToBool(value);
  } else if(key == "maxmoves") {
    rules.firstPassWin = false;
    rules.VCNRule = VCNRULE_NOVC;
    rules.maxMoves = Global::stringToInt(value);
  } else
    throw IOError("Unknown rules option: " + key);
  return rules;
}

static Rules parseRulesHelper(const string& sOrig) {
  Rules rules;
  string lowercased = Global::trim(Global::toLower(sOrig));
  if(lowercased == "chinese") {
  } else if(sOrig.length() > 0 && sOrig[0] == '{') {
    // Default if not specified
    rules = Rules::getTrompTaylorish();
    try {
      json input = json::parse(sOrig);
      string s;
      for(json::iterator iter = input.begin(); iter != input.end(); ++iter) {
        string key = iter.key();
        string v = iter.value().is_string() ? iter.value().get<string>() : to_string(iter.value());
        Rules::updateRules(key, v, rules);
        
      }
    } catch(nlohmann::detail::exception&) {
      throw IOError("Could not parse rules: " + sOrig);
    }
  }

  // This is more of a legacy internal format, not recommended for users to provide
  else {
    throw IOError("Could not parse rules: " + sOrig);
  }

  return rules;
}

Rules Rules::parseRules(const string& sOrig) {
  return parseRulesHelper(sOrig);
}

bool Rules::tryParseRules(const string& sOrig, Rules& buf) {
  Rules rules;
  try {
    rules = parseRulesHelper(sOrig);
  } catch(const StringError&) {
    return false;
  }
  buf = rules;
  return true;
}

string Rules::toStringMaybeNice() const {
  if(*this == parseRulesHelper("chinese"))
    return "chinese";
  return toString();
}

const Hash128 Rules::ZOBRIST_BASIC_RULE_HASH[3] = {
  Hash128(0x72eeccc72c82a5e7ULL, 0x0d1265e413623e2bULL),  // Based on sha256 hash of Rules::TAX_NONE
  Hash128(0x125bfe48a41042d5ULL, 0x061866b5f2b98a79ULL),  // Based on sha256 hash of Rules::TAX_SEKI
  Hash128(0xa384ece9d8ee713cULL, 0xfdc9f3b5d1f3732bULL),  // Based on sha256 hash of Rules::TAX_ALL
};
const Hash128 Rules::ZOBRIST_FIRSTPASSWIN_HASH = Hash128(0x082b14fef06c9716ULL, 0x98f5e636a9351303ULL);

const Hash128 Rules::ZOBRIST_VCNRULE_HASH_BASE = Hash128(0x0dbdfa4e0ec7459cULL, 0xcc360848cf5d7c49ULL);

const Hash128 Rules::ZOBRIST_MAXMOVES_HASH_BASE = Hash128(0x8aba00580c378fe8ULL, 0x7f6c1210e74fb440ULL);

const Hash128 Rules::ZOBRIST_PASSNUM_B_HASH_BASE = Hash128(0x5a881a894f189de8ULL, 0x80adfc5ab8789990ULL);

const Hash128 Rules::ZOBRIST_PASSNUM_W_HASH_BASE = Hash128(0x0d9c957db399f5b2ULL, 0xbf7a532d567346b6ULL);
