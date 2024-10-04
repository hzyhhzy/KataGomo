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

#ifndef COMPILE_MAX_BOARD_LEN
#define COMPILE_MAX_BOARD_LEN 13
#endif

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
  Loc getMirrorLoc(Loc loc, int x_size, int y_size);
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
  static Hash128 ZOBRIST_BOARD_HASH[MAX_ARR_SIZE][4];
  static Hash128 ZOBRIST_BOARD_HASH2[MAX_ARR_SIZE][4];
  static Hash128 ZOBRIST_PLAYER_HASH[4];
  static Hash128 ZOBRIST_KO_LOC_HASH[MAX_ARR_SIZE];
  static Hash128 ZOBRIST_KO_MARK_HASH[MAX_ARR_SIZE][4];
  static const Hash128 ZOBRIST_PASS_ENDS_PHASE;
  static const Hash128 ZOBRIST_GAME_IS_OVER;

  //Structs---------------------------------------

  //Tracks a chain/string/group of stones
  struct ChainData {
    Player owner;        //Owner of chain
    short num_locs;      //Number of stones in chain
    short num_liberties; //Number of liberties in chain
  };

  //Tracks locations for fast random selection
  /* struct PointList { */
  /*   PointList(); */
  /*   PointList(const PointList&); */
  /*   void operator=(const PointList&); */
  /*   void add(Loc); */
  /*   void remove(Loc); */
  /*   int size() const; */
  /*   Loc& operator[](int); */
  /*   bool contains(Loc loc) const; */

  /*   Loc list_[MAX_PLAY_SIZE];   //Locations in the list */
  /*   int indices_[MAX_ARR_SIZE]; //Maps location to index in the list */
  /*   int size_; */
  /* }; */

  //Move data passed back when moves are made to allow for undos
  struct MoveRecord {
    Player pla;
    Loc loc;
    Loc ko_loc;
    uint8_t capDirs; //First 4 bits indicate directions of capture, fifth bit indicates suicide
  };

  //Constructors---------------------------------
  Board();  //Create Board of size (DEFAULT_LEN,DEFAULT_LEN)
  Board(int x, int y); //Create Board of size (x,y)
  Board(const Board& other);

  Board& operator=(const Board&) = default;

  //Functions------------------------------------

  double sqrtBoardArea() const;

  //Gets the number of stones of the chain at loc. Precondition: location must be black or white.
  int getChainSize(Loc loc) const;
  //Gets the number of liberties of the chain at loc. Precondition: location must be black or white.
  int getNumLiberties(Loc loc) const;
  //Returns the number of liberties a new stone placed here would have, or max if it would be >= max.
  int getNumLibertiesAfterPlay(Loc loc, Player pla, int max) const;
  //Returns a fast lower and upper bound on the number of liberties a new stone placed here would have
  void getBoundNumLibertiesAfterPlay(Loc loc, Player pla, int& lowerBound, int& upperBound) const;
  //Gets the number of empty spaces directly adjacent to this location
  int getNumImmediateLiberties(Loc loc) const;

  //Check if moving here would be a self-capture
  bool isSuicide(Loc loc, Player pla) const;
  //Check if moving here would be an illegal self-capture
  bool isIllegalSuicide(Loc loc, Player pla, bool isMultiStoneSuicideLegal) const;
  //Check if moving here is illegal due to simple ko
  bool isKoBanned(Loc loc) const;
  //Check if moving here is legal, ignoring simple ko
  bool isLegalIgnoringKo(Loc loc, Player pla, bool isMultiStoneSuicideLegal) const;
  //Check if moving here is legal. Equivalent to isLegalIgnoringKo && !isKoBanned
  bool isLegal(Loc loc, Player pla, bool isMultiStoneSuicideLegal) const;
  //Check if this location is on the board
  bool isOnBoard(Loc loc) const;
  //Check if this location contains a simple eye for the specified player.
  bool isSimpleEye(Loc loc, Player pla) const;
  //Check if a move at this location would be a capture of an opponent group.
  bool wouldBeCapture(Loc loc, Player pla) const;
  //Check if a move at this location would be a capture in a simple ko mouth.
  bool wouldBeKoCapture(Loc loc, Player pla) const;
  Loc getKoCaptureLoc(Loc loc, Player pla) const;
  //Check if this location is adjacent to stones of the specified color
  bool isAdjacentToPla(Loc loc, Player pla) const;
  bool isAdjacentOrDiagonalToPla(Loc loc, Player pla) const;
  //Check if this location is adjacent a given chain.
  bool isAdjacentToChain(Loc loc, Loc chain) const;
  //Does this connect two pla distinct groups that are not both pass-alive and not within opponent pass-alive area either?
  bool isNonPassAliveSelfConnection(Loc loc, Player pla, Color* passAliveArea) const;
  //Is this board empty?
  bool isEmpty() const;
  //Count the number of stones on the board
  int numStonesOnBoard() const;
  int numPlaStonesOnBoard(Player pla) const;

  //Get a hash that combines the position of the board with simple ko prohibition and a player to move.
  Hash128 getSitHashWithSimpleKo(Player pla) const;

  //Lift any simple ko ban recorded on thie board due to an immediate prior ko capture.
  void clearSimpleKoLoc();
  //Directly set that there is a simple ko prohibition on this location. Note that this is not necessarily safe
  //when also using a BoardHistory, since the BoardHistory may not know about this change, or the game could be in cleanup phase, etc.
  void setSimpleKoLoc(Loc loc);

  //Sets the specified stone if possible, including overwriting existing stones.
  //Resolves any captures and/or suicides that result from setting that stone, including deletions of the stone itself.
  //Returns false if location or color were out of range.
  bool setStone(Loc loc, Color color);

  //Sets the specified stone, including overwriting existing stones, but only if doing so will
  //not result in any captures or zero liberty groups.
  //Returns false if location or color were out of range, or if would cause a zero liberty group.
  //In case of failure, will restore the position, but may result in chain ids or ordering in the board changing.
  bool setStoneFailIfNoLibs(Loc loc, Color color);
  //Same, but sets multiple stones, and only requires that the final configuration contain no zero-liberty groups.
  //If it does contain a zero liberty group, fails and returns false and leaves the board in an arbitrarily changed but valid state.
  //Also returns false if any location is specified more than once.
  bool setStonesFailIfNoLibs(std::vector<Move> placements);

  //Attempts to play the specified move. Returns true if successful, returns false if the move was illegal.
  bool playMove(Loc loc, Player pla, bool isMultiStoneSuicideLegal);

  //Plays the specified move, assuming it is legal.
  void playMoveAssumeLegal(Loc loc, Player pla);

  //Plays the specified move, assuming it is legal, and returns a MoveRecord for the move
  MoveRecord playMoveRecorded(Loc loc, Player pla);

  //Undo the move given by record. Moves MUST be undone in the order they were made.
  //Undos will NOT typically restore the precise representation in the board to the way it was. The heads of chains
  //might change, the order of the circular lists might change, etc.
  void undo(MoveRecord record);

  //Get what the position hash would be if we were to play this move and resolve captures and suicides.
  //Assumes the move is on an empty location.
  Hash128 getPosHashAfterMove(Loc loc, Player pla) const;

  //Returns true if, for a move just played at loc, the sum of the number of stones in loc's group and the sizes of the empty regions it touches
  //are greater than bound. See also https://senseis.xmp.net/?Cycle for some interesting test cases for thinking about this bound.
  //Returns false for passes.
  bool simpleRepetitionBoundGt(Loc loc, int bound) const;

  //Get a random legal move that does not fill a simple eye.
  /* Loc getRandomMCLegal(Player pla); */

  //Check if the given stone is in unescapable atari or can be put into unescapable atari.
  //WILL perform a mutable search - may alter the linked lists or heads, etc.
  bool searchIsLadderCaptured(Loc loc, bool defenderFirst, std::vector<Loc>& buf);
  bool searchIsLadderCapturedAttackerFirst2Libs(Loc loc, std::vector<Loc>& buf, std::vector<Loc>& workingMoves);
  bool anyAlive(Player pla, bool isMultiStoneSuicideLegal) const;

  //Run some basic sanity checks on the board state, throws an exception if not consistent, for testing/debugging
  void checkConsistency() const;
  //For the moment, only used in testing since it does extra consistency checks.
  //If we need a version to be used in "prod", we could make an efficient version maybe as operator==.
  bool isEqualForTesting(const Board& other, bool checkNumCaptures, bool checkSimpleKo) const;

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

  //Every chain of stones has one of its stones arbitrarily designated as the head.
  ChainData chain_data[MAX_ARR_SIZE]; //For each head stone, the chaindata for the chain under that head. Undefined otherwise.
  Loc chain_head[MAX_ARR_SIZE];       //Where is the head of this chain? Undefined if EMPTY or WALL
  Loc next_in_chain[MAX_ARR_SIZE];    //Location of next stone in chain. Circular linked list. Undefined if EMPTY or WALL

  Loc ko_loc;   //A simple ko capture was made here, making it illegal to replay here next move

  /* PointList empty_list; //List of all empty locations on board */

  Hash128 pos_hash; //A zobrist hash of the current board position (does not include ko point or player to move)

  int numBlackCaptures; //Number of b stones captured, informational and used by board history when clearing pos
  int numWhiteCaptures; //Number of w stones captured, informational and used by board history when clearing pos

  short adj_offsets[8]; //Indices 0-3: Offsets to add for adjacent points. Indices 4-7: Offsets for diagonal points. 2 and 3 are +x and +y.

  private:
  void init(int xS, int yS);
  int countHeuristicConnectionLibertiesX2(Loc loc, Player pla) const;
  bool isLibertyOf(Loc loc, Loc head) const;
  void mergeChains(Loc loc1, Loc loc2);
  int removeChain(Loc loc);
  void removeSingleStone(Loc loc);

  void addChain(Loc loc, Player pla);
  Loc addChainHelper(Loc head, Loc tailTarget, Loc loc, Color color);
  void rebuildChain(Loc loc, Player pla);
  Loc rebuildChainHelper(Loc head, Loc tailTarget, Loc loc, Color color);
  void changeSurroundingLiberties(Loc loc, Color color, int delta);

  friend std::ostream& operator<<(std::ostream& out, const Board& board);

  int findLiberties(Loc loc, std::vector<Loc>& buf, int bufStart, int bufIdx) const;
  int findLibertyGainingCaptures(Loc loc, std::vector<Loc>& buf, int bufStart, int bufIdx) const;
  bool hasLibertyGainingCaptures(Loc loc) const;

  bool isAdjacentToPlaHead(Player pla, Loc loc, Loc plaHead) const;

  bool countEmptyHelper(bool* emptyCounted, Loc initialLoc, int& count, int bound) const;

  //static void monteCarloOwner(Player player, Board* board, int mc_counts[]);
};




#endif // GAME_BOARD_H_
