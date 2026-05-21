#include "../game/board.h"
#include "../game/gamelogic.h"
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
Hash128 Board::ZOBRIST_SIZE_Z_HASH[MAX_LEN+1];
Hash128 Board::ZOBRIST_BOARD_HASH[MAX_PLAY_SIZE][NUM_BOARD_COLORS];
Hash128 Board::ZOBRIST_STAGENUM_HASH[STAGE_NUM_EACH_PLA];
Hash128 Board::ZOBRIST_STAGELOC_HASH[MAX_EXTENDED_ARR_SIZE][STAGE_NUM_EACH_PLA];
Hash128 Board::ZOBRIST_NEXTPLA_HASH[4];
Hash128 Board::ZOBRIST_PLAYER_HASH[4];
Hash128 Board::ZOBRIST_KO_LOC_HASH[MAX_EXTENDED_ARR_SIZE];
const Hash128 Board::ZOBRIST_GAME_IS_OVER = //Based on sha256 hash of Board::ZOBRIST_GAME_IS_OVER
  Hash128(0xb6f9e465597a77eeULL, 0xf1d583d960a4ce7fULL);

//LOCATION--------------------------------------------------------------------------------
Loc Location::getLoc(int x, int y, int x_size)
{
  return (Loc)(x + y*x_size);
}
Loc Location::getLoc(int x, int y, int z, int x_size, int y_size)
{
  return (Loc)(x + y*x_size + z*x_size*y_size);
}
int Location::getX(Loc loc, int x_size)
{
  return loc % x_size;
}
int Location::getY(Loc loc, int x_size)
{
  return loc / x_size;
}
int Location::getY(Loc loc, int x_size, int y_size)
{
  return (loc / x_size) % y_size;
}
int Location::getZ(Loc loc, int x_size, int y_size)
{
  return loc / (x_size*y_size);
}
int Location::getAdjacentOffsets(short adj_offsets[26], int x_size, int y_size, int z_size)
{
  int count = 0;
  adj_offsets[count++] = (short)1;
  adj_offsets[count++] = (short)-1;
  adj_offsets[count++] = (short)x_size;
  adj_offsets[count++] = (short)-x_size;
  if(z_size > 1) {
    adj_offsets[count++] = (short)(x_size*y_size);
    adj_offsets[count++] = (short)(-x_size*y_size);
  }
  return count;
}

bool Location::isAdjacent(Loc loc0, Loc loc1, int x_size)
{
  return isAdjacent(loc0,loc1,x_size,1,1);
}
bool Location::isAdjacent(Loc loc0, Loc loc1, int x_size, int y_size, int z_size)
{
  int dx = getX(loc1,x_size) - getX(loc0,x_size);
  int dy = getY(loc1,x_size,y_size) - getY(loc0,x_size,y_size);
  int dz = getZ(loc1,x_size,y_size) - getZ(loc0,x_size,y_size);
  (void)z_size;
  return std::abs(dx) + std::abs(dy) + std::abs(dz) == 1;
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


//CONSTRUCTORS AND INITIALIZATION----------------------------------------------------------

Board::Board()
{
  init(DEFAULT_LEN, DEFAULT_LEN, DEFAULT_LEN);
}

Board::Board(int x, int y)
{
  init(x,y,y);
}

Board::Board(int x, int y, int z)
{
  init(x,y,z);
}


Board::Board(const Board& other)
{
  x_size = other.x_size;
  y_size = other.y_size;
  z_size = other.z_size;
  play_size = other.play_size;

  memcpy(colors, other.colors, sizeof(Color)*MAX_PLAY_SIZE);
  memcpy(oneLibertyStones, other.oneLibertyStones, sizeof(bool)*MAX_PLAY_SIZE);

  pos_hash = other.pos_hash;

  memcpy(adj_offsets, other.adj_offsets, sizeof(short) * MAX_ADJ_OFFSETS);
  adj_offset_count = other.adj_offset_count;

  nextPla = other.nextPla;
  ko_loc = other.ko_loc;
  stage = other.stage;
  memcpy(midLocs, other.midLocs, sizeof(Loc) * STAGE_NUM_EACH_PLA);
}

void Board::init(int xS, int yS)
{
  init(xS,yS,yS);
}

void Board::init(int xS, int yS, int zS)
{
  assert(IS_ZOBRIST_INITALIZED);
  if(xS < 0 || yS < 0 || zS < 0 || xS > MAX_LEN || yS > MAX_LEN || zS > MAX_LEN)
    throw StringError("Board::init - invalid board size");

  x_size = xS;
  y_size = yS;
  z_size = zS;
  play_size = x_size * y_size * z_size;

  for(int i = 0; i < MAX_PLAY_SIZE; i++)
    colors[i] = C_WALL;
  for(int i = 0; i < MAX_PLAY_SIZE; i++)
    oneLibertyStones[i] = false;

  for(int z = 0; z < z_size; z++) {
    for(int y = 0; y < y_size; y++) {
      for(int x = 0; x < x_size; x++) {
        Loc loc = Location::getLoc(x,y,z,x_size,y_size);
        colors[loc] = C_EMPTY;
      }
    }
  }
  for(int z = z_size; z < MAX_LEN; z++) {
    (void)z;
  }
  for(int i = 0; i < STAGE_NUM_EACH_PLA; i++) {
    midLocs[i] = Board::NULL_LOC;
  }
  nextPla = C_BLACK;
  ko_loc = Board::NULL_LOC;
  stage = 0;

  pos_hash = ZOBRIST_SIZE_X_HASH[x_size] ^ ZOBRIST_SIZE_Y_HASH[y_size] ^ ZOBRIST_SIZE_Z_HASH[z_size] ^ ZOBRIST_NEXTPLA_HASH[nextPla] ^
             ZOBRIST_STAGENUM_HASH[stage];

  adj_offset_count = Location::getAdjacentOffsets(adj_offsets, x_size, y_size, z_size);
  rebuildOneLibertyTable();
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

  //Do this second so that the player and encore hashes are not
  //afffected by the size of the board we compile with.
  for(int i = 0; i<MAX_PLAY_SIZE; i++) {
    for(Color j = 0; j < NUM_BOARD_COLORS; j++) {
      if(j == C_EMPTY || j == C_WALL)
        ZOBRIST_BOARD_HASH[i][j] = Hash128();
      else
        ZOBRIST_BOARD_HASH[i][j] = nextHash();
    }
  }

  for(int i = 0; i < STAGE_NUM_EACH_PLA; i++) {
    ZOBRIST_STAGENUM_HASH[i] = nextHash();
    for(int j = 0; j < MAX_EXTENDED_ARR_SIZE; j++)
      ZOBRIST_STAGELOC_HASH[j][i] = nextHash();
    ZOBRIST_STAGELOC_HASH[Board::NULL_LOC][i] = Hash128();
  }
  ZOBRIST_STAGENUM_HASH[0] = Hash128();

  for(Color j = 0; j < 4; j++) {
    ZOBRIST_NEXTPLA_HASH[j] = nextHash();
  }

  for(int i = 0; i < MAX_EXTENDED_ARR_SIZE; i++) {
    ZOBRIST_KO_LOC_HASH[i] = nextHash();
  }
  ZOBRIST_KO_LOC_HASH[Board::NULL_LOC] = Hash128();



  //Reseed the random number generator so that these size hashes are also
  //not affected by the size of the board we compile with
  rand.init("Board::initHash() for ZOBRIST_SIZE hashes");
  for(int i = 0; i<MAX_LEN+1; i++) {
    ZOBRIST_SIZE_X_HASH[i] = nextHash();
    ZOBRIST_SIZE_Y_HASH[i] = nextHash();
    ZOBRIST_SIZE_Z_HASH[i] = nextHash();
  }


  IS_ZOBRIST_INITALIZED = true;
}


bool Board::isOnBoard(Loc loc) const {
  return loc >= 0 && loc < play_size;
}

//Check if moving here is illegal.
bool Board::isLegal(Loc loc, Player pla) const
{
  return GameLogic::isLegal(*this, pla, loc);
}

bool Board::isEmpty() const {
  for(int loc = 0; loc < play_size; loc++) {
    if(colors[loc] != C_EMPTY)
      return false;
  }
  return true;
}

int Board::numStonesOnBoard() const {
  int num = 0;
  for(int loc = 0; loc < play_size; loc++) {
    if(colors[loc] != C_EMPTY)
      num += 1;
  }
  return num;
}

int Board::numPlaStonesOnBoard(Player pla) const {
  int num = 0;
  for(int loc = 0; loc < play_size; loc++) {
    if(colors[loc] == pla)
      num += 1;
  }
  return num;
}

int Board::boardArea() const {
  return play_size - numPlaStonesOnBoard(C_BAN);
}

int Board::boardVolume() const {
  return play_size;
}

bool Board::isCubical() const {
  return x_size == y_size && y_size == z_size;
}

static void getAdjacentLocs(const Board& board, Loc loc, Loc* adjs, int& numAdjs) {
  numAdjs = 0;
  int x = Location::getX(loc, board.x_size);
  int y = Location::getY(loc, board.x_size, board.y_size);
  int z = Location::getZ(loc, board.x_size, board.y_size);
  if(x > 0)
    adjs[numAdjs++] = Location::getLoc(x-1,y,z,board.x_size,board.y_size);
  if(x+1 < board.x_size)
    adjs[numAdjs++] = Location::getLoc(x+1,y,z,board.x_size,board.y_size);
  if(y > 0)
    adjs[numAdjs++] = Location::getLoc(x,y-1,z,board.x_size,board.y_size);
  if(y+1 < board.y_size)
    adjs[numAdjs++] = Location::getLoc(x,y+1,z,board.x_size,board.y_size);
  if(z > 0)
    adjs[numAdjs++] = Location::getLoc(x,y,z-1,board.x_size,board.y_size);
  if(z+1 < board.z_size)
    adjs[numAdjs++] = Location::getLoc(x,y,z+1,board.x_size,board.y_size);
}

static void collectGroup(const Board& board, Loc loc, vector<Loc>& stones, bool* visited) {
  stones.clear();
  Color pla = board.colors[loc];
  if(pla != C_BLACK && pla != C_WHITE)
    return;
  vector<Loc> stack;
  stack.push_back(loc);
  visited[loc] = true;
  while(!stack.empty()) {
    Loc cur = stack.back();
    stack.pop_back();
    stones.push_back(cur);
    Loc adjs[6];
    int numAdjs = 0;
    getAdjacentLocs(board, cur, adjs, numAdjs);
    for(int i = 0; i < numAdjs; i++) {
      Loc adj = adjs[i];
      if(!visited[adj] && board.colors[adj] == pla) {
        visited[adj] = true;
        stack.push_back(adj);
      }
    }
  }
}

static int countGroupLiberties(const Board& board, const vector<Loc>& stones) {
  bool seenLibs[Board::MAX_PLAY_SIZE];
  std::fill(seenLibs, seenLibs + Board::MAX_PLAY_SIZE, false);
  int numLibs = 0;
  for(Loc stone: stones) {
    Loc adjs[6];
    int numAdjs = 0;
    getAdjacentLocs(board, stone, adjs, numAdjs);
    for(int i = 0; i < numAdjs; i++) {
      Loc adj = adjs[i];
      if(board.colors[adj] == C_EMPTY && !seenLibs[adj]) {
        seenLibs[adj] = true;
        numLibs += 1;
      }
    }
  }
  return numLibs;
}

static int collectGroupAndCountLibertiesUpTo(const Board& board, Loc loc, vector<Loc>& stones, bool* visited, int libertyLimit, bool stopWhenReachedLimit) {
  stones.clear();
  Color pla = board.colors[loc];
  if(pla != C_BLACK && pla != C_WHITE)
    return 0;

  bool seenLibs[Board::MAX_PLAY_SIZE];
  std::fill(seenLibs, seenLibs + Board::MAX_PLAY_SIZE, false);
  int numLibs = 0;

  vector<Loc> stack;
  stack.push_back(loc);
  visited[loc] = true;
  while(!stack.empty()) {
    Loc cur = stack.back();
    stack.pop_back();
    stones.push_back(cur);

    Loc adjs[6];
    int numAdjs = 0;
    getAdjacentLocs(board, cur, adjs, numAdjs);
    for(int i = 0; i < numAdjs; i++) {
      Loc adj = adjs[i];
      if(board.colors[adj] == pla && !visited[adj]) {
        visited[adj] = true;
        stack.push_back(adj);
      }
      else if(board.colors[adj] == C_EMPTY && numLibs < libertyLimit && !seenLibs[adj]) {
        seenLibs[adj] = true;
        numLibs += 1;
        if(stopWhenReachedLimit && numLibs >= libertyLimit)
          return numLibs;
      }
    }
  }
  return numLibs;
}

void Board::rebuildOneLibertyTable() {
  for(int loc = 0; loc < MAX_PLAY_SIZE; loc++)
    oneLibertyStones[loc] = false;

  bool visited[MAX_PLAY_SIZE];
  std::fill(visited, visited + MAX_PLAY_SIZE, false);
  vector<Loc> stones;
  for(Loc loc = 0; loc < play_size; loc++) {
    Color pla = colors[loc];
    if((pla == C_BLACK || pla == C_WHITE) && !visited[loc]) {
      int numLibs = collectGroupAndCountLibertiesUpTo(*this, loc, stones, visited, 2, false);
      if(numLibs == 1) {
        for(Loc stone: stones)
          oneLibertyStones[stone] = true;
      }
    }
  }
}

bool Board::isKoBanned(Loc loc) const {
  return loc == ko_loc;
}

bool Board::isInOneLibertyGroup(Loc loc) const {
  if(!isOnBoard(loc))
    return false;
  if(colors[loc] != C_BLACK && colors[loc] != C_WHITE)
    return false;
  return oneLibertyStones[loc];
}

bool Board::isSingleStoneSuicide(Loc loc, Player pla) const {
  return isIllegalSuicide(loc, pla, true);
}

bool Board::isIllegalSuicide(Loc loc, Player pla, bool multiStoneSuicideLegal) const {
  if(pla != C_BLACK && pla != C_WHITE)
    return false;
  if(!isOnBoard(loc) || colors[loc] != C_EMPTY)
    return false;

  Loc adjs[6];
  int numAdjs = 0;
  getAdjacentLocs(*this, loc, adjs, numAdjs);
  for(int i = 0; i < numAdjs; i++) {
    Color adjColor = colors[adjs[i]];
    if(adjColor == C_EMPTY)
      return false;
    if(adjColor == pla && (multiStoneSuicideLegal || !isInOneLibertyGroup(adjs[i])))
      return false;
    if(adjColor == getOpp(pla) && isInOneLibertyGroup(adjs[i]))
      return false;
  }
  return true;
}

int Board::countLiberties(Loc loc) const {
  if(!isOnBoard(loc) || (colors[loc] != C_BLACK && colors[loc] != C_WHITE))
    return 0;
  bool visited[MAX_PLAY_SIZE];
  std::fill(visited, visited + MAX_PLAY_SIZE, false);
  vector<Loc> stones;
  collectGroup(*this, loc, stones, visited);
  return countGroupLiberties(*this, stones);
}

int Board::getChainSize(Loc loc) const {
  if(!isOnBoard(loc) || (colors[loc] != C_BLACK && colors[loc] != C_WHITE))
    return 0;
  bool visited[MAX_PLAY_SIZE];
  std::fill(visited, visited + MAX_PLAY_SIZE, false);
  vector<Loc> stones;
  collectGroup(*this, loc, stones, visited);
  return (int)stones.size();
}

double Board::calculateAreaScoreWhiteMinusBlack(float komi) const {
  double whiteScore = komi;
  double blackScore = 0.0;
  bool visited[MAX_PLAY_SIZE];
  std::fill(visited, visited + MAX_PLAY_SIZE, false);

  for(Loc loc = 0; loc < play_size; loc++) {
    if(colors[loc] == C_WHITE) {
      whiteScore += 1.0;
      continue;
    }
    if(colors[loc] == C_BLACK) {
      blackScore += 1.0;
      continue;
    }
    if(colors[loc] != C_EMPTY || visited[loc])
      continue;

    vector<Loc> region;
    vector<Loc> stack;
    stack.push_back(loc);
    visited[loc] = true;
    bool touchesBlack = false;
    bool touchesWhite = false;
    while(!stack.empty()) {
      Loc cur = stack.back();
      stack.pop_back();
      region.push_back(cur);

      Loc adjs[6];
      int numAdjs = 0;
      getAdjacentLocs(*this, cur, adjs, numAdjs);
      for(int i = 0; i < numAdjs; i++) {
        Loc adj = adjs[i];
        if(colors[adj] == C_EMPTY && !visited[adj]) {
          visited[adj] = true;
          stack.push_back(adj);
        }
        else if(colors[adj] == C_BLACK)
          touchesBlack = true;
        else if(colors[adj] == C_WHITE)
          touchesWhite = true;
      }
    }

    if(touchesBlack && !touchesWhite)
      blackScore += (double)region.size();
    else if(touchesWhite && !touchesBlack)
      whiteScore += (double)region.size();
  }

  return whiteScore - blackScore;
}

bool Board::setStoneInternal(Loc loc, Color color, bool rebuildOneLiberty)
{
  if(loc < 0 || loc >= play_size)
    return false;

  Color colorOld = colors[loc];
  colors[loc] = color;
  pos_hash ^= ZOBRIST_BOARD_HASH[loc][colorOld];
  pos_hash ^= ZOBRIST_BOARD_HASH[loc][color];

  if(rebuildOneLiberty)
    rebuildOneLibertyTable();

  return true;
}

bool Board::setStone(Loc loc, Color color)
{
  return setStoneInternal(loc, color, true);
}

bool Board::setStones(std::vector<Move> placements) {
  std::set<Loc> locs;
  for(const Move& placement: placements) {
    if(locs.find(placement.loc) != locs.end())
      return false;
    locs.insert(placement.loc);
  }
  // First empty out all locations that we plan to set.
  // This guarantees avoiding any intermediate liberty issues.
  for(const Move& placement: placements) {
    bool suc = setStoneInternal(placement.loc, C_EMPTY, false);
    if(!suc)
      return false;
  }
  // Now set all the stones we wanted.
  for(const Move& placement: placements) {
    bool suc = setStoneInternal(placement.loc, placement.pla, false);
    if(!suc)
      return false;
  }
  rebuildOneLibertyTable();
  return true;
}

//Plays the specified move, assuming it is legal.
void Board::playMoveAssumeLegal(Loc loc, Player pla)
{
  if(pla != nextPla) {
    std::cout << "Error next player ";
  }

  if(loc == PASS_LOC) {
    ko_loc = Board::NULL_LOC;
    stage = 0;
    for(int i = 0; i < STAGE_NUM_EACH_PLA; i++) {
      pos_hash ^= ZOBRIST_STAGELOC_HASH[midLocs[i]][i];
      midLocs[i] = Board::NULL_LOC;
    }

    pos_hash ^= ZOBRIST_NEXTPLA_HASH[nextPla];
    nextPla = getOpp(nextPla);
    pos_hash ^= ZOBRIST_NEXTPLA_HASH[nextPla];

    return;
  }
  assert(isOnBoard(loc));

  if(colors[loc] != C_EMPTY)
    ASSERT_UNREACHABLE;

  if(stage != 0) {
    pos_hash ^= ZOBRIST_STAGENUM_HASH[stage];
    stage = 0;
    pos_hash ^= ZOBRIST_STAGENUM_HASH[stage];
  }
  for(int i = 0; i < STAGE_NUM_EACH_PLA; i++) {
    pos_hash ^= ZOBRIST_STAGELOC_HASH[midLocs[i]][i];
    midLocs[i] = Board::NULL_LOC;
  }

  setStoneInternal(loc, pla, false);

  Player opp = getOpp(pla);
  int numCaptured = 0;
  Loc possibleKoLoc = Board::NULL_LOC;
  Loc adjs[6];
  int numAdjs = 0;
  getAdjacentLocs(*this, loc, adjs, numAdjs);
  for(int i = 0; i < numAdjs; i++) {
    Loc adj = adjs[i];
    if(colors[adj] != opp)
      continue;
    bool visited[MAX_PLAY_SIZE];
    std::fill(visited, visited + MAX_PLAY_SIZE, false);
    vector<Loc> oppStones;
    int oppLiberties = collectGroupAndCountLibertiesUpTo(*this, adj, oppStones, visited, 1, true);
    if(oppLiberties == 0) {
      for(Loc stone: oppStones)
        setStoneInternal(stone, C_EMPTY, false);
      numCaptured += (int)oppStones.size();
      if(oppStones.size() == 1)
        possibleKoLoc = oppStones[0];
    }
  }

  {
    bool visited[MAX_PLAY_SIZE];
    std::fill(visited, visited + MAX_PLAY_SIZE, false);
    vector<Loc> ownStones;
    int ownLiberties = collectGroupAndCountLibertiesUpTo(*this, loc, ownStones, visited, 2, true);
    if(!ownStones.empty() && ownLiberties == 0) {
      for(Loc stone: ownStones)
        setStoneInternal(stone, C_EMPTY, false);
      ko_loc = Board::NULL_LOC;
    }
    else if(numCaptured == 1 && ownStones.size() == 1 && ownLiberties == 1)
      ko_loc = possibleKoLoc;
    else
      ko_loc = Board::NULL_LOC;
  }

  rebuildOneLibertyTable();

  pos_hash ^= ZOBRIST_NEXTPLA_HASH[nextPla];
  nextPla = getOpp(nextPla);
  pos_hash ^= ZOBRIST_NEXTPLA_HASH[nextPla];
}

Player Board::nextnextPla() const {
  if(stage == STAGE_NUM_EACH_PLA - 1)
    return getOpp(nextPla);
  else
    return nextPla;
}

Player Board::prevPla() const {
  if(stage == 0)
    return getOpp(nextPla);
  else
    return nextPla;
}

Hash128 Board::getSitHash(Player pla) const {
  Hash128 h = pos_hash;
  h ^= Board::ZOBRIST_PLAYER_HASH[pla];
  return h;
}

int Location::distance(Loc loc0, Loc loc1, int x_size) {
  int dx = getX(loc1,x_size) - getX(loc0,x_size);
  int dy = getY(loc1,x_size) - getY(loc0,x_size);
  return std::abs(dx) + std::abs(dy);
}

int Location::euclideanDistanceSquared(Loc loc0, Loc loc1, int x_size) {
  int dx = getX(loc1,x_size) - getX(loc0,x_size);
  int dy = getY(loc1,x_size) - getY(loc0,x_size);
  return dx*dx + dy*dy;
}

//TACTICAL STUFF--------------------------------------------------------------------


void Board::checkConsistency() const {
  const string errLabel = string("Board::checkConsistency(): ");


  vector<Loc> buf;
  Hash128 tmp_pos_hash = ZOBRIST_SIZE_X_HASH[x_size] ^ ZOBRIST_SIZE_Y_HASH[y_size] ^ ZOBRIST_SIZE_Z_HASH[z_size];
  int emptyCount = 0;
  for(Loc loc = 0; loc < MAX_PLAY_SIZE; loc++) {
    if(loc < play_size) {
      if(colors[loc] == C_EMPTY) {
        emptyCount += 1;
      } 
      else if(colors[loc] != C_WALL) {
        tmp_pos_hash ^= ZOBRIST_BOARD_HASH[loc][colors[loc]];
        tmp_pos_hash ^= ZOBRIST_BOARD_HASH[loc][C_EMPTY];
      }
      else
        throw StringError(errLabel + "C_WALL value within board legal area");
    }
    else {
      if(colors[loc] != C_WALL)
        throw StringError(errLabel + "Non-WALL value outside of board legal area");
    }
  }


  tmp_pos_hash ^= ZOBRIST_NEXTPLA_HASH[nextPla];
  tmp_pos_hash ^= ZOBRIST_STAGENUM_HASH[stage];
  for(int i = 0; i < STAGE_NUM_EACH_PLA; i++) {
    // std::cout << ZOBRIST_STAGELOC_HASH[midLocs[i]][i]<<" ";
    tmp_pos_hash ^= ZOBRIST_STAGELOC_HASH[midLocs[i]][i];
  }

  if(pos_hash != tmp_pos_hash) {
    std::cout << "Stage=" << stage << ",NextPla=" << int(nextPla) << std::endl;
    throw StringError(errLabel + "Pos hash does not match expected");
  }

  if(ko_loc != Board::NULL_LOC) {
    if(!isOnBoard(ko_loc))
      throw StringError(errLabel + "Invalid ko loc");
    if(colors[ko_loc] != C_EMPTY)
      throw StringError(errLabel + "Ko loc is not empty");
  }

  bool expectedOneLibertyStones[MAX_PLAY_SIZE];
  for(int loc = 0; loc < MAX_PLAY_SIZE; loc++)
    expectedOneLibertyStones[loc] = false;
  bool visited[MAX_PLAY_SIZE];
  std::fill(visited, visited + MAX_PLAY_SIZE, false);
  vector<Loc> stones;
  for(Loc loc = 0; loc < play_size; loc++) {
    Color pla = colors[loc];
    if((pla == C_BLACK || pla == C_WHITE) && !visited[loc]) {
      int numLibs = collectGroupAndCountLibertiesUpTo(*this, loc, stones, visited, 2, false);
      if(numLibs == 0)
        throw StringError(errLabel + "Zero-liberty group on board");
      if(numLibs == 1) {
        for(Loc stone: stones)
          expectedOneLibertyStones[stone] = true;
      }
    }
  }
  for(Loc loc = 0; loc < MAX_PLAY_SIZE; loc++) {
    if(oneLibertyStones[loc] != expectedOneLibertyStones[loc])
      throw StringError(errLabel + "oneLibertyStones does not match expected");
  }



  short tmpAdjOffsets[MAX_ADJ_OFFSETS];
  int tmpAdjOffsetCount = Location::getAdjacentOffsets(tmpAdjOffsets,x_size,y_size,z_size);
  if(tmpAdjOffsetCount != adj_offset_count)
    throw StringError(errLabel + "Corrupted adj_offset_count");
  for(int i = 0; i<adj_offset_count; i++)
    if(tmpAdjOffsets[i] != adj_offsets[i])
      throw StringError(errLabel + "Corrupted adj_offsets array");
}

bool Board::isEqualForTesting(const Board& other) const {
  checkConsistency();
  other.checkConsistency();
  if(x_size != other.x_size)
    return false;
  if(y_size != other.y_size)
    return false;
  if(z_size != other.z_size)
    return false;
  if(pos_hash != other.pos_hash)
    return false;
  for(int i = 0; i<MAX_PLAY_SIZE; i++) {
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
  case C_BAN: return '-';
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

string Location::toStringUCI(Loc loc, int x_size, int y_size) {
  if(x_size > 26 * 26)
    return toStringMach(loc, x_size);
  if(loc == Board::PASS_LOC)
    return string("0000");
  if(loc == Board::NULL_LOC)
    return string("null");
  const char* xChar = "abcdefghijklmnopqrstuvwxyz";
  int x = getX(loc, x_size);
  int y = getY(loc, x_size);
  if(x >= x_size || x < 0 || y < 0 || y >= y_size)
    return toStringMach(loc, x_size);

  char buf[128];
  if(x <= 25)
    sprintf(buf, "%c%d", xChar[x], y_size - y);
  else
    sprintf(buf, "%c%c%d", xChar[x / 26 - 1], xChar[x % 26], y_size - y);
  return string(buf);
}

string Location::toStringUCI(Loc loc, const Board& b) {
  return toStringUCI(loc, b.x_size, b.y_size);
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

bool Board::setFEN(std::string fen, Player nextPlayer) {
  if(z_size > 1)
    throw StringError("Board::setFEN only supports z_size == 1");

  auto lines = Global::split(fen, '/');
  int newYsize = lines.size();

  int newXsize = 0;
  {
    string line = lines[0];

    int x = 0;
    for(int p = 0; p < line.size(); p++) {
      char c = line[p];
      if(c >= '0' && c <= '9') {
        int emptylen = c - '0';
        for(int i = 0; i < emptylen; i++) {
          x++;
        }
      } else {
        x++;
      }
    }
    newXsize = x;
  }
  init(newXsize, newYsize);
  pos_hash ^= ZOBRIST_NEXTPLA_HASH[nextPla];
  nextPla = nextPlayer;
  pos_hash ^= ZOBRIST_NEXTPLA_HASH[nextPla];

  assert(x_size <= 9);
  for(int y = 0; y < y_size; y++) {
    string line = lines[y];

    int x = 0;
    for(int p = 0; p < line.size(); p++) {
      if(x >= x_size)
        return false;

      char c = line[p];
      if(c >= '0' && c <= '9') {
        int emptylen = c - '0';
        for(int i = 0; i < emptylen; i++) {
          if(x >= x_size)
            return false;
          setStone(Location::getLoc(x, y, x_size), C_EMPTY);
          x++;
        }
      } else {
        Color color = 
          c == 'x' ? C_BLACK : 
          c == 'o' ? C_WHITE : 
          c == '-' ? C_BAN : 
          C_WALL;
        if(color == C_WALL)
          return false;
        setStone(Location::getLoc(x, y, x_size), color);
        x++;
      }
    }
    if(x != x_size)
      return false;
  }
  return true;
}

std::string Board::getFEN() const {
  if(z_size > 1)
    throw StringError("Board::getFEN only supports z_size == 1");
  string fen;
  for(int y = 0; y < y_size; y++) {
    if(y != 0)
      fen += "/";

    int emptylen = 0;
    for(int x = 0; x < x_size; x++) {
      Color color = colors[Location::getLoc(x, y, x_size)];
      char c = 
        color == C_BLACK ? 'x' :
        color == C_WHITE ? 'o' :
        color == C_BAN ? '-' :
        color == C_EMPTY ? '.' :
        '#';
      if(c == '.')
        emptylen += 1;
      else {
        if(emptylen > 0)
          fen += to_string(emptylen);
        emptylen = 0;
        fen += c;
      }
    }
    if(emptylen > 0)
      fen += to_string(emptylen);
  }
  fen += " ";
  if(nextPla == C_BLACK)
    fen += "b";
  else
    fen += "w";
  return fen;
}

void Board::printBoard(ostream& out, const Board& board, Loc markLoc, const vector<Move>* hist) {
  if(hist != NULL)
    out << "MoveNum: " << hist->size() << " ";
  out << "HASH: " << board.pos_hash << "\n";
  bool showCoords = board.x_size <= 50 && board.y_size <= 50;
  for(int z = 0; z < board.z_size; z++) {
    if(board.z_size > 1)
      out << "Layer " << z << "\n";
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
    for(int y = 0; y < board.y_size; y++) {
      if(showCoords) {
        char buf[16];
        sprintf(buf,"%2d",board.y_size-y);
        out << buf << ' ';
      }
      for(int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,z,board.x_size,board.y_size);
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
}

ostream& operator<<(ostream& out, const Board& board) {
  Board::printBoard(out,board,Board::NULL_LOC,NULL);
  return out;
}


string Board::toStringSimple(const Board& board, char lineDelimiter) {
  string s;
  for(int z = 0; z < board.z_size; z++) {
    if(z > 0)
      s += lineDelimiter;
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,z,board.x_size,board.y_size);
        s += PlayerIO::colorToChar(board.colors[loc]);
      }
      s += lineDelimiter;
    }
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
      else if(c == 'o' || c == 'O') {
        bool suc = board.setStone(loc,P_WHITE);
        if(!suc)
          throw StringError(string("Board::parseBoard - zero-liberty group near ") + Location::toString(loc,board));
      }
      else if(c == 'x' || c == 'X') {
        bool suc = board.setStone(loc,P_BLACK);
        if(!suc)
          throw StringError(string("Board::parseBoard - zero-liberty group near ") + Location::toString(loc,board));
      }
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

