#include "../game/board.h"
#include "../forbiddenPoint/ForbiddenPointFinder.h"
/*
 * board.cpp
 * Originally from an unreleased project back in 2010, modified since.
 * Authors: brettharrison (original), David Wu (original and later modificationss).
 */

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#include "../core/rand.h"

using namespace std;

//STATIC VARS-----------------------------------------------------------------------------
bool Board::IS_ZOBRIST_INITALIZED = false;
Hash128 Board::ZOBRIST_SIZE_X_HASH[MAX_LEN+1];
Hash128 Board::ZOBRIST_SIZE_Y_HASH[MAX_LEN+1];
Hash128 Board::ZOBRIST_BOARD_HASH[MAX_ARR_SIZE][4];
Hash128 Board::ZOBRIST_PLAYER_HASH[4];
const Hash128 Board::ZOBRIST_GAME_IS_OVER = //Based on sha256 hash of Board::ZOBRIST_GAME_IS_OVER
  Hash128(0xb6f9e465597a77eeULL, 0xf1d583d960a4ce7fULL);

//LOCATION--------------------------------------------------------------------------------
Loc Location::getLoc(int x, int y, int x_size)
{
  return (x+1) + (y+1)*(x_size+1);
}
int Location::getX(Loc loc, int x_size)
{
  return (loc % (x_size+1)) - 1;
}
int Location::getY(Loc loc, int x_size)
{
  return (loc / (x_size+1)) - 1;
}
void Location::getAdjacentOffsets(short adj_offsets[8], int x_size)
{
  adj_offsets[0] = -(x_size+1);
  adj_offsets[1] = -1;
  adj_offsets[2] = 1;
  adj_offsets[3] = (x_size+1);
  adj_offsets[4] = -(x_size+1)-1;
  adj_offsets[5] = -(x_size+1)+1;
  adj_offsets[6] = (x_size+1)-1;
  adj_offsets[7] = (x_size+1)+1;
}

bool Location::isAdjacent(Loc loc0, Loc loc1, int x_size)
{
  return loc0 == loc1 - (x_size+1) || loc0 == loc1 - 1 || loc0 == loc1 + 1 || loc0 == loc1 + (x_size+1);
}

Loc Location::getMirrorLoc(Loc loc, int x_size, int y_size) {
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
    return loc;
  return getLoc(x_size-1-getX(loc,x_size),y_size-1-getY(loc,x_size),x_size);
}

Loc Location::getCenterLoc(int x_size, int y_size) {
  if(x_size % 2 == 0 || y_size % 2 == 0)
    return Board::NULL_LOC;
  return getLoc(x_size / 2, y_size / 2, x_size);
}

Loc Location::getCenterLoc(const Board& b) {
  return getCenterLoc(b.x_size,b.y_size);
}

bool Location::isCentral(Loc loc, int x_size, int y_size) {
  int x = getX(loc,x_size);
  int y = getY(loc,x_size);
  return x >= (x_size-1)/2 && x <= x_size/2 && y >= (y_size-1)/2 && y <= y_size/2;
}

bool Location::isNearCentral(Loc loc, int x_size, int y_size) {
  int x = getX(loc,x_size);
  int y = getY(loc,x_size);
  return x >= (x_size-1)/2-1 && x <= x_size/2+1 && y >= (y_size-1)/2-1 && y <= y_size/2+1;
}


#define FOREACHADJ(BLOCK) {int ADJOFFSET = -(x_size+1); {BLOCK}; ADJOFFSET = -1; {BLOCK}; ADJOFFSET = 1; {BLOCK}; ADJOFFSET = x_size+1; {BLOCK}};
#define ADJ0 (-(x_size+1))
#define ADJ1 (-1)
#define ADJ2 (1)
#define ADJ3 (x_size+1)

//CONSTRUCTORS AND INITIALIZATION----------------------------------------------------------

Board::Board()
{
  init(DEFAULT_LEN,DEFAULT_LEN);
}

Board::Board(int x, int y)
{
  init(x,y);
}


Board::Board(const Board& other)
{
  x_size = other.x_size;
  y_size = other.y_size;

  memcpy(colors, other.colors, sizeof(Color)*MAX_ARR_SIZE);

  pos_hash = other.pos_hash;
  memcpy(adj_offsets, other.adj_offsets, sizeof(short)*8);
}

void Board::init(int xS, int yS)
{
  assert(IS_ZOBRIST_INITALIZED);
  if(xS < 0 || yS < 0 || xS > MAX_LEN || yS > MAX_LEN)
    throw StringError("Board::init - invalid board size");

  x_size = xS;
  y_size = yS;

  for(int i = 0; i < MAX_ARR_SIZE; i++)
    colors[i] = C_WALL;

  for(int y = 0; y < y_size; y++)
  {
    for(int x = 0; x < x_size; x++)
    {
      Loc loc = (x+1) + (y+1)*(x_size+1);
      colors[loc] = C_EMPTY;
      // empty_list.add(loc);
    }
  }

  pos_hash = ZOBRIST_SIZE_X_HASH[x_size] ^ ZOBRIST_SIZE_Y_HASH[y_size];
  Location::getAdjacentOffsets(adj_offsets,x_size);
}

void Board::initHash()
{
  if(IS_ZOBRIST_INITALIZED)
    return;
  Rand rand("Board::initHash()");

  auto nextHash = [&rand]() {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    return Hash128(h0,h1);
  };

  for(int i = 0; i<4; i++)
    ZOBRIST_PLAYER_HASH[i] = nextHash();

  //afffected by the size of the board we compile with.
  for(int i = 0; i<MAX_ARR_SIZE; i++) {
    for(Color j = 0; j<4; j++) {
      if(j == C_EMPTY || j == C_WALL)
        ZOBRIST_BOARD_HASH[i][j] = Hash128();
      else
        ZOBRIST_BOARD_HASH[i][j] = nextHash();

    }
  }



  //Reseed the random number generator so that these size hashes are also
  //not affected by the size of the board we compile with
  rand.init("Board::initHash() for ZOBRIST_SIZE hashes");
  for(int i = 0; i<MAX_LEN+1; i++) {
    ZOBRIST_SIZE_X_HASH[i] = nextHash();
    ZOBRIST_SIZE_Y_HASH[i] = nextHash();
  }


  IS_ZOBRIST_INITALIZED = true;
}

Hash128 Board::getSitHash(Player pla) const {
  Hash128 h = pos_hash;
  h ^= Board::ZOBRIST_PLAYER_HASH[pla];
  return h;
}

bool Board::isOnBoard(Loc loc) const {
  return loc >= 0 && loc < MAX_ARR_SIZE && colors[loc] != C_WALL;
}

//Check if moving here is illegal.
bool Board::isLegal(Loc loc, Player pla, bool isMultiStoneSuicideLegal) const
{
  (void)isMultiStoneSuicideLegal;
  if(pla != P_BLACK && pla != P_WHITE)
    return false;
  if(!( loc == PASS_LOC || (
    loc >= 0 &&
    loc < MAX_ARR_SIZE &&
    (colors[loc] == C_EMPTY)
  ))) return false;
   return true;
}


MovePriority Board::getMovePriority(Player pla, Loc loc, const Rules& rule)const
{

  if (loc == PASS_LOC)return MP_NORMAL;
  if (!isLegal(loc, pla, false))return MP_ILLEGAL;


  bool isSixWinMe =
    rule.basicRule == Rules::BASICRULE_FREESTYLE ? true :
    rule.basicRule == Rules::BASICRULE_STANDARD ? false :
    rule.basicRule == Rules::BASICRULE_RENJU ? (pla == C_WHITE) :
    true;

  bool isSixWinOpp =
    rule.basicRule == Rules::BASICRULE_FREESTYLE ? true :
    rule.basicRule == Rules::BASICRULE_STANDARD ? false :
    rule.basicRule == Rules::BASICRULE_RENJU ? (pla == C_BLACK) :
    true;

  MovePriority MP = getMovePriorityAssumeLegal(pla, loc, isSixWinMe, isSixWinOpp);
  if (MP == MP_MYLIFEFOUR && rule.basicRule == Rules::BASICRULE_RENJU && pla == C_BLACK && isForbidden(loc))return MP_NORMAL;
  return MP;
}
bool Board::checkAlreadyWin(Player pla, Loc loc, const Rules& rule) const
{
  if (loc == PASS_LOC)return false;

  bool isSixWinMe =
    rule.basicRule == Rules::BASICRULE_FREESTYLE ? true :
    rule.basicRule == Rules::BASICRULE_STANDARD ? false :
    rule.basicRule == Rules::BASICRULE_RENJU ? (pla == C_WHITE) :
    true;

  bool isSixWinOpp =
    rule.basicRule == Rules::BASICRULE_FREESTYLE ? true :
    rule.basicRule == Rules::BASICRULE_STANDARD ? false :
    rule.basicRule == Rules::BASICRULE_RENJU ? (pla == C_BLACK) :
    true;

  MovePriority MP = getMovePriorityAssumeLegal(pla, loc, isSixWinMe, isSixWinOpp);
  return MP==MP_FIVE;
}
MovePriority Board::getMovePriorityAssumeLegal(Player pla, Loc loc, bool isSixWinMe, bool isSixWinOpp)const
{
  if (loc == PASS_LOC)return MP_NORMAL;
  MovePriority MP = MP_NORMAL;
  for (int i = 0; i < 4; i++)
  {
    MovePriority tmpMP = getMovePriorityOneDirectionAssumeLegal(pla, loc, isSixWinMe, isSixWinOpp, i);
    if (tmpMP < MP)MP = tmpMP;
  }
  return MP;
}
MovePriority Board::getMovePriorityOneDirectionAssumeLegal(Player pla, Loc loc, bool isSixWinMe, bool isSixWinOpp, int adjID) const
{
  assert(adjID >= 0 && adjID < 4);
  Player opp = getOpp(pla);
  short adj = adj_offsets[2*adjID];
  bool isMyLife1, isMyLife2, isOppLife1, isOppLife2;
  int myConNum = connectionLengthOneDirection(pla, loc, adj, isSixWinMe, isMyLife1) + connectionLengthOneDirection(pla, loc, -adj, isSixWinMe, isMyLife2) + 1;
  int oppConNum = connectionLengthOneDirection(opp, loc, adj, isSixWinOpp, isOppLife1) + connectionLengthOneDirection(opp, loc, -adj, isSixWinOpp, isOppLife2) + 1;
  if (myConNum == 5 || (myConNum > 5 && isSixWinMe))return MP_FIVE;
  if (oppConNum == 5 || (oppConNum > 5 && isSixWinOpp))return MP_OPPOFOUR;


  if (myConNum == 4 && isMyLife1&&isMyLife2)return MP_MYLIFEFOUR;
  return MP_NORMAL;

}

int Board::connectionLengthOneDirection(Player pla, Loc loc, short adj, bool isSixWin, bool& isLife)const
{
  Loc tmploc = loc;
  int conNum = 0;
  isLife = false;
  while (1)
  {
    tmploc += adj;
    if (!isOnBoard(tmploc))break;
    if (colors[tmploc] == pla)conNum++;
    else if (colors[tmploc] == C_EMPTY)
    {
      isLife = true;
      if (!isSixWin)
      {

        tmploc += adj;
        if (isOnBoard(tmploc) && colors[tmploc] == pla)isLife = false;
      }
      break;
    }
    else break;
  }
  return conNum;



}
bool Board::isEmpty() const {
  for(int y = 0; y < y_size; y++) {
    for(int x = 0; x < x_size; x++) {
      Loc loc = Location::getLoc(x,y,x_size);
      if(colors[loc] != C_EMPTY)
        return false;
    }
  }
  return true;
}

int Board::numStonesOnBoard() const {
  int num = 0;
  for(int y = 0; y < y_size; y++) {
    for(int x = 0; x < x_size; x++) {
      Loc loc = Location::getLoc(x,y,x_size);
      if(colors[loc] == C_BLACK || colors[loc] == C_WHITE)
        num += 1;
    }
  }
  return num;
}

int Board::numPlaStonesOnBoard(Player pla) const {
  int num = 0;
  for(int y = 0; y < y_size; y++) {
    for(int x = 0; x < x_size; x++) {
      Loc loc = Location::getLoc(x,y,x_size);
      if(colors[loc] == pla)
        num += 1;
    }
  }
  return num;
}

bool Board::setStone(Loc loc, Color color)
{
  if(loc < 0 || loc >= MAX_ARR_SIZE || colors[loc] == C_WALL)
    return false;
  if(color != C_BLACK && color != C_WHITE && color != C_EMPTY)
    return false;

  if(colors[loc] == color)
  {}
  else if(colors[loc] == C_EMPTY)
    playMoveAssumeLegal(loc,color);
  else if(color == C_EMPTY)
    removeSingleStone(loc);
  else {
    removeSingleStone(loc);
    playMoveAssumeLegal(loc,color);
  }

  return true;
}


//Attempts to play the specified move. Returns true if successful, returns false if the move was illegal.
bool Board::playMove(Loc loc, Player pla, bool isMultiStoneSuicideLegal)
{
  if(isLegal(loc,pla,isMultiStoneSuicideLegal))
  {
    playMoveAssumeLegal(loc,pla);
    return true;
  }
  return false;
}

Hash128 Board::getPosHashAfterMove(Loc loc, Player pla) const {
  if(loc == PASS_LOC)
    return pos_hash;
  assert(loc != NULL_LOC);

  Hash128 hash = pos_hash;
  hash ^= ZOBRIST_BOARD_HASH[loc][pla];


  return hash;
}

//Plays the specified move, assuming it is legal.
void Board::playMoveAssumeLegal(Loc loc, Player pla)
{
  //Pass?
  if(loc == PASS_LOC)
  {
    return;
  }


  colors[loc] = pla;
  pos_hash ^= ZOBRIST_BOARD_HASH[loc][pla];
}

//Remove a single stone, even a stone part of a larger group.
void Board::removeSingleStone(Loc loc)
{
  Player pla = colors[loc];

  colors[loc] = C_EMPTY;
  pos_hash ^= ZOBRIST_BOARD_HASH[loc][pla];
}
bool Board::isForbidden(Loc loc) const
{
  if (loc == PASS_LOC)
  {
    return false;
  }
  if (!(loc >= 0 &&
    loc < MAX_ARR_SIZE &&
    (colors[loc] == C_EMPTY)))return false;
  if (x_size == y_size)
  {
    int x = Location::getX(loc, x_size);
    int y = Location::getY(loc, x_size);
    int nearbyBlack = 0;
    //x++; y++;
    for (int i = std::max(x - 2, 0); i <= std::min(x + 2, x_size - 1); i++)
      for (int j = std::max(y - 2, 0); j <= std::min(y + 2, y_size - 1); j++)
      {
        int xd = i - x;
        int yd = j - y;
        xd = xd > 0 ? xd : -xd;
        yd = yd > 0 ? yd : -yd;
        if (((xd + yd) != 3) && (colors[Location::getLoc(i, j, x_size)] == C_BLACK))nearbyBlack++;
      }

    if (nearbyBlack >= 2)
    {
      CForbiddenPointFinder fpf(x_size);
      for (int x = 0; x < x_size; x++)
        for (int y = 0; y < y_size; y++)
        {
          fpf.SetStone(x, y, colors[Location::getLoc(x, y, x_size)]);
        }
      if (fpf.isForbiddenNoNearbyCheck(Location::getX(loc, x_size), Location::getY(loc, x_size)))return true;
    }
  }
  return false;
}
bool Board::isForbiddenAlreadyPlayed(Loc loc) const
{
  if (loc == PASS_LOC)
  {
    return false;
  }
  if (!(loc >= 0 &&
    loc < MAX_ARR_SIZE &&
    (colors[loc] == C_BLACK)))return false;
  if (x_size == y_size)
  {
    int x = Location::getX(loc, x_size);
    int y = Location::getY(loc, x_size);
    int nearbyBlack = 0;
    //x++; y++;
    for (int i = std::max(x - 2, 0); i <= std::min(x + 2, x_size - 1); i++)
      for (int j = std::max(y - 2, 0); j <= std::min(y + 2, y_size - 1); j++)
      {
        int xd = i - x;
        int yd = j - y;
        xd = xd > 0 ? xd : -xd;
        yd = yd > 0 ? yd : -yd;
        if (((xd + yd) != 3) && (colors[Location::getLoc(i, j, x_size)] == C_BLACK))nearbyBlack++;
      }

    if (nearbyBlack >= 3)
    {
      CForbiddenPointFinder fpf(x_size);
      for (int x = 0; x < x_size; x++)
        for (int y = 0; y < y_size; y++)
        {
          fpf.SetStone(x, y, colors[Location::getLoc(x, y, x_size)]);
        }
      fpf.SetStone(Location::getX(loc, x_size), Location::getY(loc, x_size), C_EMPTY);
      if (fpf.isForbiddenNoNearbyCheck(Location::getX(loc, x_size), Location::getY(loc, x_size)))return true;
    }
  }
  return false;
}


int Location::distance(Loc loc0, Loc loc1, int x_size) {
  int dx = getX(loc1,x_size) - getX(loc0,x_size);
  int dy = (loc1-loc0-dx) / (x_size+1);
  return (dx >= 0 ? dx : -dx) + (dy >= 0 ? dy : -dy);
}

int Location::euclideanDistanceSquared(Loc loc0, Loc loc1, int x_size) {
  int dx = getX(loc1,x_size) - getX(loc0,x_size);
  int dy = (loc1-loc0-dx) / (x_size+1);
  return dx*dx + dy*dy;
}

//TACTICAL STUFF--------------------------------------------------------------------


void Board::checkConsistency() const {
  const string errLabel = string("Board::checkConsistency(): ");


  vector<Loc> buf;
  Hash128 tmp_pos_hash = ZOBRIST_SIZE_X_HASH[x_size] ^ ZOBRIST_SIZE_Y_HASH[y_size];
  for(Loc loc = 0; loc < MAX_ARR_SIZE; loc++) {
    int x = Location::getX(loc,x_size);
    int y = Location::getY(loc,x_size);
    if(x < 0 || x >= x_size || y < 0 || y >= y_size) {
      if(colors[loc] != C_WALL)
        throw StringError(errLabel + "Non-WALL value outside of board legal area");
    }
    else {
      if(colors[loc] == C_BLACK || colors[loc] == C_WHITE) {
        tmp_pos_hash ^= ZOBRIST_BOARD_HASH[loc][colors[loc]];
        tmp_pos_hash ^= ZOBRIST_BOARD_HASH[loc][C_EMPTY];
      }
      else if(colors[loc] == C_EMPTY) {
        // if(!empty_list.contains(loc))
        //   throw StringError(errLabel + "Empty list doesn't contain empty location");
      }
      else
        throw StringError(errLabel + "Non-(black,white,empty) value within board legal area");
    }
  }
  if(pos_hash != tmp_pos_hash)
    throw StringError(errLabel + "Pos hash does not match expected");


  short tmpAdjOffsets[8];
  Location::getAdjacentOffsets(tmpAdjOffsets,x_size);
  for(int i = 0; i<8; i++)
    if(tmpAdjOffsets[i] != adj_offsets[i])
      throw StringError(errLabel + "Corrupted adj_offsets array");
}

bool Board::isEqualForTesting(const Board& other, bool checkNumCaptures, bool checkSimpleKo) const {
  (void)checkNumCaptures;
  (void)checkSimpleKo;
  checkConsistency();
  other.checkConsistency();
  if(x_size != other.x_size)
    return false;
  if(y_size != other.y_size)
    return false;
  if(pos_hash != other.pos_hash)
    return false;
  for(int i = 0; i<MAX_ARR_SIZE; i++) {
    if(colors[i] != other.colors[i])
      return false;
  }
  //We don't require that the chain linked lists are in the same order.
  //Consistency check ensures that all the linked lists are consistent with colors array, which we checked.
  return true;
}



//IO FUNCS------------------------------------------------------------------------------------------

char PlayerIO::colorToChar(Color c)
{
  switch(c) {
  case C_BLACK: return 'X';
  case C_WHITE: return 'O';
  case C_EMPTY: return '.';
  default:  return '#';
  }
}

string PlayerIO::playerToString(Color c)
{
  switch(c) {
  case C_BLACK: return "Black";
  case C_WHITE: return "White";
  case C_EMPTY: return "Empty";
  default:  return "Wall";
  }
}

string PlayerIO::playerToStringShort(Color c)
{
  switch(c) {
  case C_BLACK: return "B";
  case C_WHITE: return "W";
  case C_EMPTY: return "E";
  default:  return "";
  }
}

bool PlayerIO::tryParsePlayer(const string& s, Player& pla) {
  string str = Global::toLower(s);
  if(str == "black" || str == "b") {
    pla = P_BLACK;
    return true;
  }
  else if(str == "white" || str == "w") {
    pla = P_WHITE;
    return true;
  }
  return false;
}

Player PlayerIO::parsePlayer(const string& s) {
  Player pla = C_EMPTY;
  bool suc = tryParsePlayer(s,pla);
  if(!suc)
    throw StringError("Could not parse player: " + s);
  return pla;
}

string Location::toStringMach(Loc loc, int x_size)
{
  if(loc == Board::PASS_LOC)
    return string("pass");
  if(loc == Board::NULL_LOC)
    return string("null");
  char buf[128];
  sprintf(buf,"(%d,%d)",getX(loc,x_size),getY(loc,x_size));
  return string(buf);
}

string Location::toString(Loc loc, int x_size, int y_size)
{
  if(x_size > 25*25)
    return toStringMach(loc,x_size);
  if(loc == Board::PASS_LOC)
    return string("pass");
  if(loc == Board::NULL_LOC)
    return string("null");
  const char* xChar = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
  int x = getX(loc,x_size);
  int y = getY(loc,x_size);
  if(x >= x_size || x < 0 || y < 0 || y >= y_size)
    return toStringMach(loc,x_size);

  char buf[128];
  if(x <= 24)
    sprintf(buf,"%c%d",xChar[x],y_size-y);
  else
    sprintf(buf,"%c%c%d",xChar[x/25-1],xChar[x%25],y_size-y);
  return string(buf);
}

string Location::toString(Loc loc, const Board& b) {
  return toString(loc,b.x_size,b.y_size);
}

string Location::toStringMach(Loc loc, const Board& b) {
  return toStringMach(loc,b.x_size);
}

static bool tryParseLetterCoordinate(char c, int& x) {
  if(c >= 'A' && c <= 'H')
    x = c-'A';
  else if(c >= 'a' && c <= 'h')
    x = c-'a';
  else if(c >= 'J' && c <= 'Z')
    x = c-'A'-1;
  else if(c >= 'j' && c <= 'z')
    x = c-'a'-1;
  else
    return false;
  return true;
}

bool Location::tryOfString(const string& str, int x_size, int y_size, Loc& result) {
  string s = Global::trim(str);
  if(s.length() < 2)
    return false;
  if(Global::isEqualCaseInsensitive(s,string("pass")) || Global::isEqualCaseInsensitive(s,string("pss"))) {
    result = Board::PASS_LOC;
    return true;
  }
  if(s[0] == '(') {
    if(s[s.length()-1] != ')')
      return false;
    s = s.substr(1,s.length()-2);
    vector<string> pieces = Global::split(s,',');
    if(pieces.size() != 2)
      return false;
    int x;
    int y;
    bool sucX = Global::tryStringToInt(pieces[0],x);
    bool sucY = Global::tryStringToInt(pieces[1],y);
    if(!sucX || !sucY)
      return false;
    result = Location::getLoc(x,y,x_size);
    return true;
  }
  else {
    int x;
    if(!tryParseLetterCoordinate(s[0],x))
      return false;

    //Extended format
    if((s[1] >= 'A' && s[1] <= 'Z') || (s[1] >= 'a' && s[1] <= 'z')) {
      int x1;
      if(!tryParseLetterCoordinate(s[1],x1))
        return false;
      x = (x+1) * 25 + x1;
      s = s.substr(2,s.length()-2);
    }
    else {
      s = s.substr(1,s.length()-1);
    }

    int y;
    bool sucY = Global::tryStringToInt(s,y);
    if(!sucY)
      return false;
    y = y_size - y;
    if(x < 0 || y < 0 || x >= x_size || y >= y_size)
      return false;
    result = Location::getLoc(x,y,x_size);
    return true;
  }
}

bool Location::tryOfStringAllowNull(const string& str, int x_size, int y_size, Loc& result) {
  if(str == "null") {
    result = Board::NULL_LOC;
    return true;
  }
  return tryOfString(str, x_size, y_size, result);
}

bool Location::tryOfString(const string& str, const Board& b, Loc& result) {
  return tryOfString(str,b.x_size,b.y_size,result);
}

bool Location::tryOfStringAllowNull(const string& str, const Board& b, Loc& result) {
  return tryOfStringAllowNull(str,b.x_size,b.y_size,result);
}

Loc Location::ofString(const string& str, int x_size, int y_size) {
  Loc result;
  if(tryOfString(str,x_size,y_size,result))
    return result;
  throw StringError("Could not parse board location: " + str);
}

Loc Location::ofStringAllowNull(const string& str, int x_size, int y_size) {
  Loc result;
  if(tryOfStringAllowNull(str,x_size,y_size,result))
    return result;
  throw StringError("Could not parse board location: " + str);
}

Loc Location::ofString(const string& str, const Board& b) {
  return ofString(str,b.x_size,b.y_size);
}


Loc Location::ofStringAllowNull(const string& str, const Board& b) {
  return ofStringAllowNull(str,b.x_size,b.y_size);
}

vector<Loc> Location::parseSequence(const string& str, const Board& board) {
  vector<string> pieces = Global::split(Global::trim(str),' ');
  vector<Loc> locs;
  for(size_t i = 0; i<pieces.size(); i++) {
    string piece = Global::trim(pieces[i]);
    if(piece.length() <= 0)
      continue;
    locs.push_back(Location::ofString(piece,board));
  }
  return locs;
}

void Board::printBoard(ostream& out, const Board& board, Loc markLoc, const vector<Move>* hist) {
  if(hist != NULL)
    out << "MoveNum: " << hist->size() << " ";
  out << "HASH: " << board.pos_hash << "\n";
  bool showCoords = board.x_size <= 50 && board.y_size <= 50;
  if(showCoords) {
    const char* xChar = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
    out << "  ";
    for(int x = 0; x < board.x_size; x++) {
      if(x <= 24) {
        out << " ";
        out << xChar[x];
      }
      else {
        out << "A" << xChar[x-25];
      }
    }
    out << "\n";
  }

  for(int y = 0; y < board.y_size; y++)
  {
    if(showCoords) {
      char buf[16];
      sprintf(buf,"%2d",board.y_size-y);
      out << buf << ' ';
    }
    for(int x = 0; x < board.x_size; x++)
    {
      Loc loc = Location::getLoc(x,y,board.x_size);
      char s = PlayerIO::colorToChar(board.colors[loc]);
      if(board.colors[loc] == C_EMPTY && markLoc == loc)
        out << '@';
      else
        out << s;

      bool histMarked = false;
      if(hist != NULL) {
        size_t start = hist->size() >= 3 ? hist->size()-3 : 0;
        for(size_t i = 0; start+i < hist->size(); i++) {
          if((*hist)[start+i].loc == loc) {
            out << (1+i);
            histMarked = true;
            break;
          }
        }
      }

      if(x < board.x_size-1 && !histMarked)
        out << ' ';
    }
    out << "\n";
  }
  out << "\n";
}

ostream& operator<<(ostream& out, const Board& board) {
  Board::printBoard(out,board,Board::NULL_LOC,NULL);
  return out;
}


string Board::toStringSimple(const Board& board, char lineDelimiter) {
  string s;
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      s += PlayerIO::colorToChar(board.colors[loc]);
    }
    s += lineDelimiter;
  }
  return s;
}

Board Board::parseBoard(int xSize, int ySize, const string& s) {
  return parseBoard(xSize,ySize,s,'\n');
}

Board Board::parseBoard(int xSize, int ySize, const string& s, char lineDelimiter) {
  Board board(xSize,ySize);
  vector<string> lines = Global::split(Global::trim(s),lineDelimiter);

  //Throw away coordinate labels line if it exists
  if(lines.size() == ySize+1 && Global::isPrefix(lines[0],"A"))
    lines.erase(lines.begin());

  if(lines.size() != ySize)
    throw StringError("Board::parseBoard - string has different number of board rows than ySize");

  for(int y = 0; y<ySize; y++) {
    string line = Global::trim(lines[y]);
    //Throw away coordinates if they exist
    size_t firstNonDigitIdx = 0;
    while(firstNonDigitIdx < line.length() && Global::isDigit(line[firstNonDigitIdx]))
      firstNonDigitIdx++;
    line.erase(0,firstNonDigitIdx);
    line = Global::trim(line);

    if(line.length() != xSize && line.length() != 2*xSize-1)
      throw StringError("Board::parseBoard - line length not compatible with xSize");

    for(int x = 0; x<xSize; x++) {
      char c;
      if(line.length() == xSize)
        c = line[x];
      else
        c = line[x*2];

      Loc loc = Location::getLoc(x,y,board.x_size);
      if(c == '.' || c == ' ' || c == '*' || c == ',' || c == '`')
        continue;
      else if(c == 'o' || c == 'O')
        board.setStone(loc,P_WHITE);
      else if(c == 'x' || c == 'X')
        board.setStone(loc,P_BLACK);
      else
        throw StringError(string("Board::parseBoard - could not parse board character: ") + c);
    }
  }
  return board;
}

nlohmann::json Board::toJson(const Board& board) {
  nlohmann::json data;
  data["xSize"] = board.x_size;
  data["ySize"] = board.y_size;
  data["stones"] = Board::toStringSimple(board,'|');
  return data;
}

Board Board::ofJson(const nlohmann::json& data) {
  int xSize = data["xSize"].get<int>();
  int ySize = data["ySize"].get<int>();
  Board board = Board::parseBoard(xSize,ySize,data["stones"].get<string>(),'|');
  return board;
}

