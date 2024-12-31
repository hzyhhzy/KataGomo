#pragma once
#include "NNUEBoard.h"
#include "Eva_nnuev2.h"
#include "../game/rules.h"
#include "../game/boardhistory.h"
using namespace NNUE;
using namespace NNUEV2;
class NNUEBoard
{
public:
  Board board;

  Rules rules;

  Eva_nnuev2 blackEvaluator;
  Eva_nnuev2 whiteEvaluator;
  float gfInputBuf[NNUEV2::globalFeatureNum];
  bool illegalMapBuf[MaxBS * MaxBS];



  NNUEBoard(const ModelWeight* weights);
  ~NNUEBoard() {}

  void clear(const Board& b, const Rules& r);

  //void setStone(Color color, Loc loc);
  void play(Color color, Loc loc);
  void undo(const Board& oldBoard, Color color, Loc loc);

  void updateInputBuf(Color nextPlayer);//gfInputBuf and illegalMapBuf

  NNUE::ValueType evaluateFull(Color color, NNUE::PolicyType* policy);
  void evaluatePolicy(Color color, NNUE::PolicyType* policy);
  NNUE::ValueType evaluateValue(Color color);



  
  //Color* board() const { return blackEvaluator->board; }

private:

  //ÿ�ε���play����undoʱ���Ȳ���EvaluatorOneSide�����ߣ���Ϊ�����ܴ��Ȼ��档
  //MCTS��ʱ�򣬾������߻�ͷ·����ʹ��cache�������١�
  struct MoveCache
  {
    bool isUndo;
    Color color;
    Loc loc;
    MoveCache() :isUndo(false), color(C_EMPTY), loc(Board::NULL_LOC){}
    MoveCache(bool isUndo,Color color,Loc loc) :isUndo(isUndo), color(color), loc(loc){}
  };

  MoveCache moveCacheB[MaxBS * MaxBS], moveCacheW[MaxBS * MaxBS];
  int moveCacheBlength, moveCacheWlength;

  void clearCache(Color color);//�����л���Ĳ�����գ�ʹ��evaluatorOneSide��board�������board��ͬ
  void addCache(bool isUndo, Color color, Loc loc);
  bool isContraryMove(MoveCache a, MoveCache b);//�ǲ��ǿ��Ե�����һ�Բ���
};

