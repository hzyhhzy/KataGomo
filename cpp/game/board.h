/*
 * board.h
 * Originally from an unreleased project back in 2010, modified since.
 * Authors: brettharrison (original), David Wu (original and later modifications).
 */

#ifndef GAME_BOARD_H_
#define GAME_BOARD_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../external/nlohmann_json/json.hpp"

#ifdef COMPILE_MAX_BOARD_LEN
static_assert(COMPILE_MAX_BOARD_LEN should not be defined);
#endif
#define COMPILE_MAX_BOARD_LEN 15

//how many stages in each move
//eg: Chess has 2 stages: moving which piece, and where to place.
static const int STAGE_NUM_EACH_PLA = 3;

//max moves num of a game
static const int MAX_MOVE_NUM = 100 * COMPILE_MAX_BOARD_LEN * COMPILE_MAX_BOARD_LEN;


//TYPES AND CONSTANTS-----------------------------------------------------------------

struct Board;

//Player
typedef int8_t Player;
static constexpr Player P_BLACK = 1;
static constexpr Player P_WHITE = 2;

//Color of a point on the board
typedef int8_t Color;
static constexpr Color C_EMPTY = 0;
static constexpr Color C_BLACK = 1;
static constexpr Color C_WHITE = 2;
static constexpr Color C_WALL = 3;
static constexpr int NUM_BOARD_COLORS = 4;

static inline Color getOpp(Color c)
{return c ^ 3;}

//Conversions for players and colors
namespace PlayerIO {
  char colorToChar(Color c);
  std::string playerToStringShort(Player p);
  std::string playerToString(Player p);
  bool tryParsePlayer(const std::string& s, Player& pla);
  Player parsePlayer(const std::string& s);
}

//Location of a point on the board
//(x,y) is represented as (x+1) + (y+1)*(x_size+1)
typedef short Loc;
namespace Location
{
  Loc getLoc(int x, int y, int x_size);
  int getX(Loc loc, int x_size);
  int getY(Loc loc, int x_size);

  void getAdjacentOffsets(short adj_offsets[8], int x_size);
  bool isAdjacent(Loc loc0, Loc loc1, int x_size);
  Loc getCenterLoc(int x_size, int y_size);
  Loc getCenterLoc(const Board& b);
  bool isCentral(Loc loc, int x_size, int y_size);
  bool isNearCentral(Loc loc, int x_size, int y_size);
  int distance(Loc loc0, Loc loc1, int x_size);
  int euclideanDistanceSquared(Loc loc0, Loc loc1, int x_size);

  std::string toString(Loc loc, int x_size, int y_size);
  std::string toString(Loc loc, const Board& b);
  std::string toStringMach(Loc loc, int x_size);
  std::string toStringMach(Loc loc, const Board& b);

  bool tryOfString(const std::string& str, int x_size, int y_size, Loc& result);
  bool tryOfString(const std::string& str, const Board& b, Loc& result);
  Loc ofString(const std::string& str, int x_size, int y_size);
  Loc ofString(const std::string& str, const Board& b);

  //Same, but will parse "null" as Board::NULL_LOC
  bool tryOfStringAllowNull(const std::string& str, int x_size, int y_size, Loc& result);
  bool tryOfStringAllowNull(const std::string& str, const Board& b, Loc& result);
  Loc ofStringAllowNull(const std::string& str, int x_size, int y_size);
  Loc ofStringAllowNull(const std::string& str, const Board& b);

  std::vector<Loc> parseSequence(const std::string& str, const Board& b);
}

//Simple structure for storing moves. Not used below, but this is a convenient place to define it.
STRUCT_NAMED_PAIR(Loc,loc,Player,pla,Move);

//Fast lightweight board designed for playouts and simulations, where speed is essential.
//Simple ko rule only.
//Does not enforce player turn order.

class ConnectedBlock {
  public:
  int16_t local_xsize;             // Width of this block's bounding box
  int16_t local_ysize;             // Height of this block's bounding box
  int16_t start_x;             // bias on the original board
  int16_t start_y;             // bias on the original board
  std::vector<bool> is_stone;  // Flattened grid: true if stone of the component is present

  ConnectedBlock(int w, int h, int x, int y);

  void setStoneRelativeToOrigin(int local_x, int local_y);

  // "isStoneRelativeToOrigin没必要判断是否越界" - the current implementation does this correctly.
  // If local_x or local_y are out of bounds [0, size-1], it returns false.
  // This is desired because a reflection landing outside the bounding box means no symmetric partner *within the
  // block*.
  bool isStoneRelativeToOrigin(int local_x, int local_y) const;

  int stoneNum() const;

  // Helper to get all actual stone points in local coordinates for iteration
  // This is still useful to avoid iterating over all cells in is_stone multiple times.
  std::vector<std::pair<int, int>> getLocalStoneCoordinates() const;

  bool isSymmetric() const;
};

// The getConnectedBlocks function from the previous response remains largely the same,
// as its job is to find components and map them into the ConnectedBlock's bounding box representation.
// For completeness, here it is again (assuming Board, Loc, Player, Color, Location namespace are defined):


struct Board
{
  //Initialization------------------------------
  //Initialize the zobrist hash.
  //MUST BE CALLED AT PROGRAM START!
  static void initHash();

  //Board parameters and Constants----------------------------------------

  static constexpr int MAX_LEN = COMPILE_MAX_BOARD_LEN;  //Maximum edge length allowed for the board
  static constexpr int DEFAULT_LEN = std::min(MAX_LEN,19); //Default edge length for board if unspecified
  static constexpr int MAX_PLAY_SIZE = MAX_LEN * MAX_LEN;  //Maximum number of playable spaces
  static constexpr int MAX_ARR_SIZE = (MAX_LEN+1)*(MAX_LEN+2)+1; //Maximum size of arrays needed

  //Location used to indicate an invalid spot on the board.
  static constexpr Loc NULL_LOC = 0;
  //Location used to indicate a pass move is desired.
  static constexpr Loc PASS_LOC = 1;

  //Zobrist Hashing------------------------------
  static bool IS_ZOBRIST_INITALIZED;
  static Hash128 ZOBRIST_SIZE_X_HASH[MAX_LEN+1];
  static Hash128 ZOBRIST_SIZE_Y_HASH[MAX_LEN+1];
  static Hash128 ZOBRIST_BOARD_HASH[MAX_ARR_SIZE][NUM_BOARD_COLORS];
  static Hash128 ZOBRIST_STAGENUM_HASH[STAGE_NUM_EACH_PLA];
  static Hash128 ZOBRIST_STAGELOC_HASH[MAX_ARR_SIZE][STAGE_NUM_EACH_PLA];
  static Hash128 ZOBRIST_NEXTPLA_HASH[4];
  static Hash128 ZOBRIST_MOVENUM_HASH[MAX_MOVE_NUM];
  static Hash128 ZOBRIST_PLAYER_HASH[4];
  static Hash128 ZOBRIST_LARGETOWERPROTECT_HASH[4];
  static const Hash128 ZOBRIST_GAME_IS_OVER;

  //Structs---------------------------------------

  //Constructors---------------------------------
  Board();  //Create Board of size (DEFAULT_LEN,DEFAULT_LEN)
  Board(int x, int y); //Create Board of size (x,y)
  Board(const Board& other);

  Board& operator=(const Board&) = default;

  //Functions------------------------------------

  bool isLegal(Loc loc, Player pla) const;
  //Check if this location is on the board
  bool isOnBoard(Loc loc) const;
  //Is this board empty?
  bool isEmpty() const;
  //Count the number of stones on the board
  int numStonesOnBoard() const;
  int numPlaStonesOnBoard(Player pla) const;


  //Sets the specified stone if possible, including overwriting existing stones.
  //Resolves any captures and/or suicides that result from setting that stone, including deletions of the stone itself.
  //Returns false if location or color were out of range.
  bool setStone(Loc loc, Color color);

  // Same, but sets multiple stones, and only requires that the final configuration contain no zero-liberty groups.
  // If it does contain a zero liberty group, fails and returns false and leaves the board in an arbitrarily changed but
  // valid state. Also returns false if any location is specified more than once.
  bool setStones(std::vector<Move> placements);

  //Plays the specified move, assuming it is legal.
  void playMoveAssumeLegal(Loc loc, Player pla);

  // who plays the next next move
  Player nextnextPla() const;

  // who plays the last move
  Player prevPla() const;

  //is attacking opp's large tower?
  bool isMoveAttackLargeTower(Color opp, Loc src, Loc dst) const;  // should be called before movement


  
  Hash128 getSitHash(Player pla) const;
  

  //Run some basic sanity checks on the board state, throws an exception if not consistent, for testing/debugging
  void checkConsistency() const;
  //For the moment, only used in testing since it does extra consistency checks.
  //If we need a version to be used in "prod", we could make an efficient version maybe as operator==.
  bool isEqualForTesting(const Board& other) const;

  static Board parseBoard(int xSize, int ySize, const std::string& s);
  static Board parseBoard(int xSize, int ySize, const std::string& s, char lineDelimiter);
  static void printBoard(std::ostream& out, const Board& board, Loc markLoc, const std::vector<Move>* hist);
  static std::string toStringSimple(const Board& board, char lineDelimiter);
  static nlohmann::json toJson(const Board& board);
  static Board ofJson(const nlohmann::json& data);

  //Data--------------------------------------------

  int x_size;                  //Horizontal size of board
  int y_size;                  //Vertical size of board
  Color colors[MAX_ARR_SIZE];  //Color of each location on the board.
  int movenum; //how many moves

  /* PointList empty_list; //List of all empty locations on board */

  Hash128 pos_hash; //A zobrist hash of the current board position (does not include ko point or player to move)

  short adj_offsets[8]; //Indices 0-3: Offsets to add for adjacent points. Indices 4-7: Offsets for diagonal points. 2 and 3 are +x and +y.

  
  //which stage. Normally 0 = choosing piece. 1 = where to place
  int stage;

  //who plays the next move
  Color nextPla;

  //一步内每一阶段的选点
  //例如：象棋类midLoc[0]是选择的棋子，midLoc[1]是落点
  Loc midLocs[STAGE_NUM_EACH_PLA];

  bool largeTowerProtect[2];  // 第一个index是color-1
  //对称棋的棋块信息
  std::vector<ConnectedBlock> connectedBlocks[2];
  int16_t smallTowerCount[2];              // 第一个index是color-1
  int8_t largeTowerInfo[2][MAX_ARR_SIZE];//0默认，1在大塔内，2在大塔周围一圈，第一个index是color-1


  private:
  void init(int xS, int yS);

  void updateConnectedBlocks(Color pla);

  friend std::ostream& operator<<(std::ostream& out, const Board& board);

  void swapNextPlayer(bool attackLargeTower);

  //static void monteCarloOwner(Player player, Board* board, int mc_counts[]);
};




#endif // GAME_BOARD_H_
