#include "../game/rules.h"

#include "../external/nlohmann_json/json.hpp"

#include <sstream>

using namespace std;
using json = nlohmann::json;

Rules::Rules() {
  //Defaults if not set - closest match to TT rules
  scoringRule = SCORING_R0;
}

Rules::Rules(
  int sRule
)
  :scoringRule(sRule)
{}

Rules::~Rules() {
}

bool Rules::operator==(const Rules& other) const {
  return
    scoringRule == other.scoringRule;
}

bool Rules::operator!=(const Rules& other) const {
  return
    scoringRule != other.scoringRule ;
}


Rules Rules::getTrompTaylorish() {
  Rules rules;
  rules.scoringRule = SCORING_R0;
  return rules;
}



set<string> Rules::scoringRuleStrings() {
  return {"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"};
}

int Rules::parseScoringRule(const string& s) {
  if(s == "R0")
    return Rules::SCORING_R0;
  else if(s == "R1")
    return Rules::SCORING_R1;
  else if(s == "R2")
    return Rules::SCORING_R2;
  else if(s == "R3")
    return Rules::SCORING_R3;
  else if(s == "R4")
    return Rules::SCORING_R4;
  else if(s == "R5")
    return Rules::SCORING_R5;
  else if(s == "R6")
    return Rules::SCORING_R6;
  else if(s == "R7")
    return Rules::SCORING_R7;
  else throw IOError("Rules::parseScoringRule: Invalid scoring rule: " + s);
}

string Rules::writeScoringRule(int scoringRule) {
  if(scoringRule == Rules::SCORING_R0) return string("R0");
  else if(scoringRule == Rules::SCORING_R1) return string("R1");
  else if(scoringRule == Rules::SCORING_R2) return string("R2");
  else if(scoringRule == Rules::SCORING_R3) return string("R3");
  else if(scoringRule == Rules::SCORING_R4) return string("R4");
  else if(scoringRule == Rules::SCORING_R5) return string("R5");
  else if(scoringRule == Rules::SCORING_R6) return string("R6");
  else if(scoringRule == Rules::SCORING_R7) return string("R7");
  return string("UNKNOWN");
}

ostream& operator<<(ostream& out, const Rules& rules) {
  out << "score" << Rules::writeScoringRule(rules.scoringRule);
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
  return ret;
}


Rules Rules::updateRules(const string& k, const string& v, Rules oldRules) {
  Rules rules = oldRules;
  string key = Global::trim(k);
  string value = Global::trim(Global::toUpper(v));
  if(key == "score") rules.scoringRule = Rules::parseScoringRule(value);
  else if(key == "scoring") rules.scoringRule = Rules::parseScoringRule(value);
  else throw IOError("Unknown rules option: " + key);
  return rules;
}

static Rules parseRulesHelper(const string& sOrig) {
  Rules rules;
  string lowercased = Global::trim(Global::toLower(sOrig));
  
  if(lowercased == "tromp-taylor" || lowercased == "tromp_taylor" || lowercased == "tromp taylor" || lowercased == "tromptaylor") {
    rules.scoringRule = Rules::SCORING_R0;
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
        else
          throw IOError("Unknown rules option: " + key);
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

      if(startsWithAndStrip(s,"scoring")) {
        if(startsWithAndStrip(s,"R0")) rules.scoringRule = Rules::SCORING_R0;
        else if(startsWithAndStrip(s,"R1")) rules.scoringRule = Rules::SCORING_R1;
        else if(startsWithAndStrip(s,"R2")) rules.scoringRule = Rules::SCORING_R2;
        else if(startsWithAndStrip(s,"R3")) rules.scoringRule = Rules::SCORING_R3;
        else if(startsWithAndStrip(s,"R4")) rules.scoringRule = Rules::SCORING_R4;
        else if(startsWithAndStrip(s,"R5")) rules.scoringRule = Rules::SCORING_R5;
        else if(startsWithAndStrip(s,"R6")) rules.scoringRule = Rules::SCORING_R6;
        else if(startsWithAndStrip(s,"R7")) rules.scoringRule = Rules::SCORING_R7;
        else throw IOError("Could not parse rules: " + sOrig);
        continue;
      }
      if(startsWithAndStrip(s,"score")) {
        if(startsWithAndStrip(s,"R0")) rules.scoringRule = Rules::SCORING_R0;
        else if(startsWithAndStrip(s,"R1")) rules.scoringRule = Rules::SCORING_R1;
        else if(startsWithAndStrip(s,"R2")) rules.scoringRule = Rules::SCORING_R2;
        else if(startsWithAndStrip(s,"R3")) rules.scoringRule = Rules::SCORING_R3;
        else if(startsWithAndStrip(s,"R4")) rules.scoringRule = Rules::SCORING_R4;
        else if(startsWithAndStrip(s,"R5")) rules.scoringRule = Rules::SCORING_R5;
        else if(startsWithAndStrip(s,"R6")) rules.scoringRule = Rules::SCORING_R6;
        else if(startsWithAndStrip(s,"R7")) rules.scoringRule = Rules::SCORING_R7;
        else throw IOError("Could not parse rules: " + sOrig);
        continue;
      }

      //Unknown rules format
      else throw IOError("Could not parse rules: " + sOrig);
    }
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




const Hash128 Rules::ZOBRIST_SCORING_RULE_HASH[24] = {
  //Based on sha256 hash of Rules::SCORING_AREA, but also mixing none tax rule hash, to preserve legacy hashes
  Hash128(0x8b3ed7598f901494ULL ^ 0x72eeccc72c82a5e7ULL, 0x1dfd47ac77bce5f8ULL ^ 0x0d1265e413623e2bULL),
  //Based on sha256 hash of Rules::SCORING_TERRITORY, but also mixing seki tax rule hash, to preserve legacy hashes
  Hash128(0x381345dc357ec982ULL ^ 0x125bfe48a41042d5ULL, 0x03ba55c026026b56ULL ^ 0x061866b5f2b98a79ULL),
  Hash128(0x4827641b70bc2559ULL, 0x3db4ebb5ea9ded1dULL),
  Hash128(0xd3e9dbd41f5dbb80ULL, 0xf6544bc012ced602ULL),
  Hash128(0xa5e77d6780c8e24aULL, 0x9a1a088f89d7db71ULL),
  Hash128(0x53409c0edd7189abULL, 0x5c3162bb1821fb80ULL),
  Hash128(0x6966857dac287033ULL, 0xd2b734976d4e8149ULL),
  Hash128(0x636d39194ffe605dULL, 0xb1b8b342364cc2c0ULL),
  Hash128(0xc07da22ad2c95160ULL, 0xa408d2614380d353ULL),
  Hash128(0x155a5a19f32ee202ULL, 0xab528887bb234f59ULL),
  Hash128(0xc1a5dcb699801ea8ULL, 0x52c798d196b452a0ULL),
  Hash128(0xf353ec471be019e8ULL, 0x7006e464e7955819ULL),
  Hash128(0x987c9450c0e00b95ULL, 0x0a74e360766322d9ULL),
  Hash128(0x1084271c12c6405aULL, 0xcb8d5127807bec36ULL),
  Hash128(0x0a6a13bc2761728eULL, 0xf0b0c57e7b0298edULL),
  Hash128(0x3462a6838506ea24ULL, 0x95e513308988d6d9ULL),
  Hash128(0x2fb38f67baabf09dULL, 0xf5416869978f437fULL),
  Hash128(0x8d171e809463e9c0ULL, 0xf0190548df9dc847ULL),
  Hash128(0xfd527cc8c65a8966ULL, 0x899cab106738dd39ULL),
  Hash128(0x4a913924cc8e41ccULL, 0x7e3a0e454a5e69fcULL),
  Hash128(0x7d8b9ec154a3db1bULL, 0x653ba4c38cab21c4ULL),
  Hash128(0xc0f335fc92da9da1ULL, 0x5abe8f578bceb081ULL),
  Hash128(0x65505f1e377fe0bcULL, 0xabc840998bce2dd4ULL),
  Hash128(0xa0a9845071b6cd15ULL, 0xa728e4f143bec316ULL),
};