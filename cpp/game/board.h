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
#define COMPILE_MAX_BOARD_LEN 9  // 9x7 board

// how many stages in each move
// eg: Chess has 2 stages: moving which piece, and where to place.
static const int STAGE_NUM_EACH_PLA = 2;

// max moves num of a game
static const int MAX_MOVE_NUM = 100 * COMPILE_MAX_BOARD_LEN * COMPILE_MAX_BOARD_LEN;

// TYPES AND CONSTANTS-----------------------------------------------------------------

struct Board;

// Player
typedef int8_t Player;
static constexpr Player P_EMPTY = 0;
static constexpr Player P_BLACK = 1;
static constexpr Player P_WHITE = 2;
static constexpr Player P_NONE = 3;

// Color of a point on the board
typedef int8_t Color;

// first 4 colors are reserved, to avoid potential bugs
static constexpr Color C_EMPTY = 0;
static constexpr Color C_BLACK = 1;
static constexpr Color C_WHITE = 2;
static constexpr Color C_WALL = 3;

// no color pieces
static constexpr Color C_RAT = 0x1;       // lv1 rat
static constexpr Color C_CAT = 0x2;       // lv2 cat
static constexpr Color C_DOG = 0x3;       // lv3 dog
static constexpr Color C_WOLF = 0x4;      // lv4 wolf
static constexpr Color C_LEOPARD = 0x5;   // lv5 leopard
static constexpr Color C_TIGER = 0x6;     // lv6 tiger
static constexpr Color C_LION = 0x7;      // lv7 lion
static constexpr Color C_ELEPHANT = 0x8;  // lv8 elephant

// static constexpr Color C_TRAP = 0x9;   // trap around home
// static constexpr Color C_HOME = 0x10;   // game end when enter opponent's home

// color = 16*player + piece
// eg: black king =  1 * 16 + 1 = 0x11,   white pawn = 2 * 16 + 7 = 0x27

// static constexpr Color C_RIVER = 0x11; //river, river does not have color

static constexpr int NUM_BOARD_COLORS = 0x30;

static inline Color getOpp(Color c) {
  return c ^ 3;
}

static_assert(NUM_BOARD_COLORS == 0x30, "This depend on getPiece,getPiecePla,getPieceType");
static inline Color getPiece(Player p, Color piece) {
  return (p << 4) | piece;
}
static inline Player getPiecePla(Color c) {
  return c >> 4;
}
static inline Color getPieceType(Color c) {
  return c & 0xF;
}

//Conversions for players and colors
namespace PlayerIO {
  char colorToChar(Color c);
  Color charToColor(char c);
  std::string playerToStringShort(Player p);
  std::string playerToString(Player p);
  bool tryParsePlayer(const std::string& s, Player& pla);
  Player parsePlayer(const std::string& s);
}

//Location of a point on the board
//(x,y) is represented as (x+1) + (y+1)*(x_size+1)
typedef short Loc;
namespace Location {
  constexpr Loc getLocConst(int x, int y) {
    return (x + 1) + (y + 1) * (7 + 1);
  }
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

struct Board
{
  //Initialization------------------------------
  //Initialize the zobrist hash.
  //MUST BE CALLED AT PROGRAM START!
  static void initHash();

  //Board parameters and Constants----------------------------------------

  static constexpr int MAX_LEN = COMPILE_MAX_BOARD_LEN;  //Maximum edge length allowed for the board
  static constexpr int DEFAULT_LEN_X = 7; //Default edge length for board if unspecified
  static constexpr int DEFAULT_LEN_Y = 9; //Default edge length for board if unspecified
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
  static Hash128 ZOBRIST_MOVENUMSLC_HASH[MAX_MOVE_NUM];
  static Hash128 ZOBRIST_PLAYER_HASH[4];

  static Hash128 ZOBRIST_MM_RULE_HASH[MAX_MOVE_NUM];
  static Hash128 ZOBRIST_MC_RULE_HASH[MAX_MOVE_NUM];

  static Hash128 ZOBRIST_73RULE_LOC_HASH[MAX_ARR_SIZE][4];
  static Hash128 ZOBRIST_73RULE_HISTORY_HASH[MAX_ARR_SIZE][7][4];

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

  bool maybeCrossRiver(Loc loc0, Loc loc1) const;


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


  
  Hash128 getSitHash(Player pla) const;
  

  //Run some basic sanity checks on the board state, throws an exception if not consistent, for testing/debugging
  void checkConsistency() const;
  //For the moment, only used in testing since it does extra consistency checks.
  //If we need a version to be used in "prod", we could make an efficient version maybe as operator==.
  bool isEqualForTesting(const Board& other) const;

  bool setFEN(std::string fen, Player nextPlayer);
  bool setFEN(std::string fen);
  std::string getFEN() const;

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
  int movenumslc; //how many moves since last capture

  /* PointList empty_list; //List of all empty locations on board */

  Hash128 pos_hash; //A zobrist hash of the current board position (does not include ko point or player to move)

  short adj_offsets[8]; //Indices 0-3: Offsets to add for adjacent points. Indices 4-7: Offsets for diagonal points. 2 and 3 are +x and +y.

  
  //which stage. Normally 0 = choosing piece. 1 = where to place
  int stage;

  //who plays the next move
  Color nextPla;

  //һ����ÿһ�׶ε�ѡ��
  //���磺������midLoc[0]��ѡ������ӣ�midLoc[1]�����
  Loc midLocs[STAGE_NUM_EACH_PLA];


  private:
  void init(int xS, int yS);


  friend std::ostream& operator<<(std::ostream& out, const Board& board);


  //static void monteCarloOwner(Player player, Board* board, int mc_counts[]);
};




#endif // GAME_BOARD_H_
