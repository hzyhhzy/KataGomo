#include "../neuralnet/nninputs.h"

using namespace std;

int NNPos::xyToPos(int x, int y, int nnXLen) {
  return y * nnXLen + x;
}
int NNPos::locToPos(Loc loc, int nnLen) {
  if(loc == Board::PASS_LOC)
    return nnLen;
  else if(loc == Board::NULL_LOC)
    return nnLen + 1;
  return loc;
}
int NNPos::locToPos(Loc loc, int boardXSize, int nnXLen, int nnYLen) {
  (void)boardXSize;
  return locToPos(loc, nnXLen * nnYLen);
}
int NNPos::locToPos(Loc loc, int boardXSize, int boardYSize, int boardZSize, int nnXLen, int nnYLen, int nnZLen) {
  if(loc == Board::PASS_LOC || loc == Board::NULL_LOC)
    return locToPos(loc, nnXLen * nnYLen * nnZLen);
  int x = Location::getX(loc, boardXSize);
  int y = Location::getY(loc, boardXSize, boardYSize);
  int z = Location::getZ(loc, boardXSize, boardYSize);
  assert(x >= 0 && x < nnXLen);
  assert(y >= 0 && y < nnYLen);
  assert(z >= 0 && z < nnZLen);
  (void)boardZSize;
  return x + y * nnXLen + z * nnXLen * nnYLen;
}
Loc NNPos::posToLoc(int pos, int boardVolume, int nnLen) {
  if(pos == nnLen)
    return Board::PASS_LOC;
  if(pos < 0 || pos >= boardVolume)
    return Board::NULL_LOC;
  return (Loc)pos;
}
Loc NNPos::posToLoc(int pos, int boardXSize, int boardYSize, int nnXLen, int nnYLen) {
  return posToLoc(pos, boardXSize * boardYSize, nnXLen * nnYLen);
}
Loc NNPos::posToLoc(int pos, int boardXSize, int boardYSize, int boardZSize, int nnXLen, int nnYLen, int nnZLen) {
  int nnLen = nnXLen * nnYLen * nnZLen;
  if(pos == nnLen)
    return Board::PASS_LOC;
  if(pos < 0 || pos >= nnLen)
    return Board::NULL_LOC;

  int x = pos % nnXLen;
  int y = (pos / nnXLen) % nnYLen;
  int z = pos / (nnXLen * nnYLen);
  if(x >= boardXSize || y >= boardYSize || z >= boardZSize)
    return Board::NULL_LOC;
  return Location::getLoc(x, y, z, boardXSize, boardYSize);
}

bool NNPos::isPassPos(int pos, int nnLen) {
  return pos == nnLen;
}

bool NNPos::isPassPos(int pos, int nnXLen, int nnYLen) {
  return isPassPos(pos, nnXLen * nnYLen);
}
bool NNPos::isPassPos(int pos, int nnXLen, int nnYLen, int nnZLen) {
  return isPassPos(pos, nnXLen * nnYLen * nnZLen);
}

int NNPos::getPolicySize(int nnLen) {
  return nnLen + 1;
}

int NNPos::getPolicySize(int nnXLen, int nnYLen) {
  return getPolicySize(nnXLen * nnYLen);
}
int NNPos::getPolicySize(int nnXLen, int nnYLen, int nnZLen) {
  return getPolicySize(nnXLen * nnYLen * nnZLen);
}

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

const Hash128 MiscNNInputParams::ZOBRIST_PLAYOUT_DOUBLINGS =
  Hash128(0xa5e6114d380bfc1dULL, 0x4160557f1222f4adULL);
const Hash128 MiscNNInputParams::ZOBRIST_NN_POLICY_TEMP =
  Hash128(0xebcbdfeec6f4334bULL, 0xb85e43ee243b5ad2ULL);

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

double ScoreValue::whiteWinsOfWinner(Player winner, double noResultUtilityForWhite) {
  if(winner == P_WHITE)
    return 1.0;
  else if(winner == P_BLACK)
    return 0.0;

  assert(winner == C_EMPTY);
  return noResultUtilityForWhite;
}

static const double twoOverPi = 0.63661977236758134308;
static const double piOverTwo = 1.57079632679489661923;


NNOutput::NNOutput()
  :nnLen(0),nnXLen(0),nnYLen(0),nnZLen(0),noisedPolicyProbs(NULL)
{}
NNOutput::NNOutput(const NNOutput& other) {
  nnHash = other.nnHash;
  whiteWinProb = other.whiteWinProb;
  whiteLossProb = other.whiteLossProb;
  whiteNoResultProb = other.whiteNoResultProb;
  varTimeLeft = other.varTimeLeft;
  shorttermWinlossError = other.shorttermWinlossError;

  nnLen = other.nnLen;
  nnXLen = other.nnXLen;
  nnYLen = other.nnYLen;
  nnZLen = other.nnZLen;

  if(other.noisedPolicyProbs != NULL) {
    noisedPolicyProbs = new float[NNPos::MAX_NN_POLICY_SIZE];
    std::copy(other.noisedPolicyProbs, other.noisedPolicyProbs + NNPos::MAX_NN_POLICY_SIZE, noisedPolicyProbs);
  }
  else
    noisedPolicyProbs = NULL;

  std::copy(other.policyProbs, other.policyProbs+NNPos::MAX_NN_POLICY_SIZE, policyProbs);
}

NNOutput::NNOutput(const vector<shared_ptr<NNOutput>>& others) {
  assert(others.size() < 1000000);
  int len = (int)others.size();
  float floatLen = (float)len;
  assert(len > 0);
  for(int i = 1; i<len; i++) {
    assert(others[i]->nnHash == others[0]->nnHash);
  }
  nnHash = others[0]->nnHash;

  whiteWinProb = 0.0f;
  whiteLossProb = 0.0f;
  whiteNoResultProb = 0.0f;
  varTimeLeft = 0.0f;
  shorttermWinlossError = 0.0f;
  for(int i = 0; i<len; i++) {
    const NNOutput& other = *(others[i]);
    whiteWinProb += other.whiteWinProb;
    whiteLossProb += other.whiteLossProb;
    whiteNoResultProb += other.whiteNoResultProb;
    varTimeLeft += other.varTimeLeft;
    shorttermWinlossError += other.shorttermWinlossError;
  }
  whiteWinProb /= floatLen;
  whiteLossProb /= floatLen;
  whiteNoResultProb /= floatLen;
  varTimeLeft /= floatLen;
  shorttermWinlossError /= floatLen;

  nnLen = others[0]->nnLen;
  nnXLen = others[0]->nnXLen;
  nnYLen = others[0]->nnYLen;
  nnZLen = others[0]->nnZLen;

  noisedPolicyProbs = NULL;

  //For technical correctness in case of impossibly rare hash collisions:
  //Just give up if they don't all match in move legality
  {
    bool mismatch = false;
    std::fill(policyProbs, policyProbs + NNPos::MAX_NN_POLICY_SIZE, 0.0f);
    for(int i = 0; i<len; i++) {
      const NNOutput& other = *(others[i]);
      for(int pos = 0; pos<NNPos::MAX_NN_POLICY_SIZE; pos++) {
        if(i > 0 && (policyProbs[pos] < 0) != (other.policyProbs[pos] < 0))
          mismatch = true;
        policyProbs[pos] += other.policyProbs[pos];
      }
    }
    //In case of mismatch, just take the first one
    //This should basically never happen, only on true hash collisions
    if(mismatch) {
      const NNOutput& other = *(others[0]);
      std::copy(other.policyProbs, other.policyProbs + NNPos::MAX_NN_POLICY_SIZE, policyProbs);
    }
    else {
      for(int pos = 0; pos<NNPos::MAX_NN_POLICY_SIZE; pos++)
        policyProbs[pos] /= floatLen;
    }
  }

}

NNOutput& NNOutput::operator=(const NNOutput& other) {
  if(&other == this)
    return *this;
  nnHash = other.nnHash;
  whiteWinProb = other.whiteWinProb;
  whiteLossProb = other.whiteLossProb;
  whiteNoResultProb = other.whiteNoResultProb;
  varTimeLeft = other.varTimeLeft;
  shorttermWinlossError = other.shorttermWinlossError;

  nnLen = other.nnLen;
  nnXLen = other.nnXLen;
  nnYLen = other.nnYLen;
  nnZLen = other.nnZLen;

  if(noisedPolicyProbs != NULL)
    delete[] noisedPolicyProbs;
  if(other.noisedPolicyProbs != NULL) {
    noisedPolicyProbs = new float[NNPos::MAX_NN_POLICY_SIZE];
    std::copy(other.noisedPolicyProbs, other.noisedPolicyProbs + NNPos::MAX_NN_POLICY_SIZE, noisedPolicyProbs);
  }
  else
    noisedPolicyProbs = NULL;

  std::copy(other.policyProbs, other.policyProbs+NNPos::MAX_NN_POLICY_SIZE, policyProbs);

  return *this;
}


NNOutput::~NNOutput() {
  if(noisedPolicyProbs != NULL) {
    delete[] noisedPolicyProbs;
    noisedPolicyProbs = NULL;
  }
}


void NNOutput::debugPrint(ostream& out, const Board& board) {
  out << "Win " << Global::strprintf("%.2fc",whiteWinProb*100) << endl;
  out << "Loss " << Global::strprintf("%.2fc",whiteLossProb*100) << endl;
  out << "NoResult " << Global::strprintf("%.2fc",whiteNoResultProb*100) << endl;
  out << "VarTimeLeft " << Global::strprintf("%.1f",varTimeLeft) << endl;
  out << "STWinlossError " << Global::strprintf("%.3f",shorttermWinlossError) << endl;

  out << "Policy" << endl;
  for(int z = 0; z<board.z_size; z++) {
    if(board.z_size > 1)
      out << "Layer " << z << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::locToPos(Location::getLoc(x,y,z,board.x_size,board.y_size), nnLen);
        float prob = policyProbs[pos];
        if(prob < 0)
          out << "   - ";
        else
          out << Global::strprintf("%4d ", (int)round(prob * 1000));
      }
      out << endl;
    }
  }
}

//-------------------------------------------------------------------------------------------------------------

namespace {
  static const int SYM_PERMUTATIONS[6][3] = {
    {0,1,2},
    {0,2,1},
    {1,0,2},
    {1,2,0},
    {2,0,1},
    {2,1,0},
  };

  static inline void decode3DSymmetry(int symmetry, int perm[3], bool flip[3]) {
    assert(symmetry >= 0 && symmetry < SymmetryHelpers::NUM_SYMMETRIES);
    int permIdx = SymmetryHelpers::getPermutationIndex(symmetry);
    for(int i = 0; i < 3; i++)
      perm[i] = SYM_PERMUTATIONS[permIdx][i];
    flip[0] = SymmetryHelpers::isFlipX(symmetry);
    flip[1] = SymmetryHelpers::isFlipY(symmetry);
    flip[2] = SymmetryHelpers::isFlipZ(symmetry);
  }

  static inline void apply3DSymmetry(int x, int y, int z, int xSize, int ySize, int zSize, int symmetry, int& outX, int& outY, int& outZ) {
    int perm[3];
    bool flip[3];
    decode3DSymmetry(symmetry, perm, flip);
    int src[3] = {x,y,z};
    int srcDims[3] = {xSize,ySize,zSize};
    int dst[3];
    for(int i = 0; i < 3; i++) {
      int value = src[perm[i]];
      int axisLen = srcDims[perm[i]];
      if(flip[i])
        value = axisLen - value - 1;
      dst[i] = value;
    }
    outX = dst[0];
    outY = dst[1];
    outZ = dst[2];
  }

  static inline int encode3DSymmetry(const int perm[3], const bool flip[3]) {
    int permIdx = -1;
    for(int i = 0; i < 6; i++) {
      if(
        SYM_PERMUTATIONS[i][0] == perm[0] &&
        SYM_PERMUTATIONS[i][1] == perm[1] &&
        SYM_PERMUTATIONS[i][2] == perm[2]
      ) {
        permIdx = i;
        break;
      }
    }
    assert(permIdx >= 0);
    return
      (permIdx << 3) |
      (flip[0] ? 0x1 : 0) |
      (flip[1] ? 0x2 : 0) |
      (flip[2] ? 0x4 : 0);
  }

  static inline int invert3DSymmetryFast(int symmetry) {
    int perm[3];
    bool flip[3];
    decode3DSymmetry(symmetry, perm, flip);

    int inversePerm[3];
    bool inverseFlip[3];
    for(int i = 0; i < 3; i++) {
      inversePerm[perm[i]] = i;
      inverseFlip[perm[i]] = flip[i];
    }
    return encode3DSymmetry(inversePerm, inverseFlip);
  }

  static inline int compose3DSymmetryFast(int firstSymmetry, int nextSymmetry) {
    int firstPerm[3];
    bool firstFlip[3];
    decode3DSymmetry(firstSymmetry, firstPerm, firstFlip);

    int nextPerm[3];
    bool nextFlip[3];
    decode3DSymmetry(nextSymmetry, nextPerm, nextFlip);

    int composedPerm[3];
    bool composedFlip[3];
    for(int i = 0; i < 3; i++) {
      composedPerm[i] = firstPerm[nextPerm[i]];
      composedFlip[i] = nextFlip[i] != firstFlip[nextPerm[i]];
    }
    return encode3DSymmetry(composedPerm, composedFlip);
  }

  static inline int getCubeSymPos(int pos, int len, int symmetry) {
    int layerArea = len * len;
    int z = pos / layerArea;
    int rem = pos % layerArea;
    int y = rem / len;
    int x = rem % len;
    int symX, symY, symZ;
    apply3DSymmetry(x,y,z,len,len,len,symmetry,symX,symY,symZ);
    return symX + symY * len + symZ * layerArea;
  }

  static void copyWithSymmetry2D(const float* src, float* dst, int nSize, int hSize, int wSize, int cSize, bool useNHWC, int symmetry, bool reverse) {
    bool transpose = (symmetry & 0x4) != 0 && hSize == wSize;
    bool flipX = (symmetry & 0x2) != 0;
    bool flipY = (symmetry & 0x1) != 0;
    if(transpose && !reverse)
      std::swap(flipX,flipY);
    if(useNHWC) {
      int nStride = hSize * wSize * cSize;
      int hStride = wSize * cSize;
      int wStride = cSize;
      int hBaseNew = 0; int hStrideNew = hStride;
      int wBaseNew = 0; int wStrideNew = wStride;

      if(flipY) { hBaseNew = (hSize-1) * hStrideNew; hStrideNew = -hStrideNew; }
      if(flipX) { wBaseNew = (wSize-1) * wStrideNew; wStrideNew = -wStrideNew; }

      if(transpose)
        std::swap(hStrideNew,wStrideNew);

      for(int n = 0; n<nSize; n++) {
        for(int h = 0; h<hSize; h++) {
          int nhOld = n * nStride + h*hStride;
          int nhNew = n * nStride + hBaseNew + h*hStrideNew;
          for(int w = 0; w<wSize; w++) {
            int nhwOld = nhOld + w*wStride;
            int nhwNew = nhNew + wBaseNew + w*wStrideNew;
            for(int c = 0; c<cSize; c++) {
              dst[nhwNew + c] = src[nhwOld + c];
            }
          }
        }
      }
    }
    else {
      int ncSize = nSize * cSize;
      int ncStride = hSize * wSize;
      int hStride = wSize;
      int wStride = 1;
      int hBaseNew = 0; int hStrideNew = hStride;
      int wBaseNew = 0; int wStrideNew = wStride;

      if(flipY) { hBaseNew = (hSize-1) * hStrideNew; hStrideNew = -hStrideNew; }
      if(flipX) { wBaseNew = (wSize-1) * wStrideNew; wStrideNew = -wStrideNew; }

      if(transpose)
        std::swap(hStrideNew,wStrideNew);

      for(int nc = 0; nc<ncSize; nc++) {
        for(int h = 0; h<hSize; h++) {
          int nchOld = nc * ncStride + h*hStride;
          int nchNew = nc * ncStride + hBaseNew + h*hStrideNew;
          for(int w = 0; w<wSize; w++) {
            int nchwOld = nchOld + w*wStride;
            int nchwNew = nchNew + wBaseNew + w*wStrideNew;
            dst[nchwNew] = src[nchwOld];
          }
        }
      }
    }
  }

  static void copyWithSymmetry(
    const float* src,
    float* dst,
    int nSize,
    int hSize,
    int wSize,
    int zSize,
    int cSize,
    bool useNHWC,
    int symmetry,
    bool reverse) {
    int totalSize = hSize * wSize * zSize;
    int appliedSymmetry = reverse ? SymmetryHelpers::invert(symmetry) : symmetry;
    if(useNHWC) {
      int nStride = totalSize * cSize;
      for(int n = 0; n < nSize; n++) {
        for(int z = 0; z < zSize; z++) {
          for(int y = 0; y < hSize; y++) {
            for(int x = 0; x < wSize; x++) {
              int pos = x + y * wSize + z * wSize * hSize;
              int symX, symY, symZ;
              apply3DSymmetry(x,y,z,wSize,hSize,zSize,appliedSymmetry,symX,symY,symZ);
              int symPos = symX + symY * wSize + symZ * wSize * hSize;
              for(int c = 0; c < cSize; c++)
                dst[n * nStride + symPos * cSize + c] = src[n * nStride + pos * cSize + c];
            }
          }
        }
      }
    }
    else {
      int channelStride = totalSize;
      for(int n = 0; n < nSize; n++) {
        for(int c = 0; c < cSize; c++) {
          int base = (n * cSize + c) * channelStride;
          for(int z = 0; z < zSize; z++) {
            for(int y = 0; y < hSize; y++) {
              for(int x = 0; x < wSize; x++) {
                int pos = x + y * wSize + z * wSize * hSize;
                int symX, symY, symZ;
                apply3DSymmetry(x,y,z,wSize,hSize,zSize,appliedSymmetry,symX,symY,symZ);
                int symPos = symX + symY * wSize + symZ * wSize * hSize;
                dst[base + symPos] = src[base + pos];
              }
            }
          }
        }
      }
    }
  }
}


void SymmetryHelpers::copyInputsWithSymmetry(
  const float* src,
  float* dst,
  int nSize,
  int hSize,
  int wSize,
  int zSize,
  int cSize,
  bool useNHWC,
  int symmetry) {
  copyWithSymmetry(src, dst, nSize, hSize, wSize, zSize, cSize, useNHWC, symmetry, false);
}

void SymmetryHelpers::copyOutputsWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int zSize, int symmetry) {
  copyWithSymmetry(src, dst, nSize, hSize, wSize, zSize, 1, false, symmetry, true);
}

int SymmetryHelpers::invert(int symmetry) {
  return invert3DSymmetryFast(symmetry);
}

int SymmetryHelpers::compose(int firstSymmetry, int nextSymmetry) {
  return compose3DSymmetryFast(firstSymmetry, nextSymmetry);
}

int SymmetryHelpers::compose(int firstSymmetry, int nextSymmetry, int nextNextSymmetry) {
  return compose(compose(firstSymmetry,nextSymmetry),nextNextSymmetry);
}

Loc SymmetryHelpers::getSymLoc(int x, int y, int xSize, int ySize, int symmetry) {
  if(symmetry < 8) {
    bool transpose = (symmetry & 0x4) != 0;
    bool flipX = (symmetry & 0x2) != 0;
    bool flipY = (symmetry & 0x1) != 0;
    if(flipX) { x = xSize - x - 1; }
    if(flipY) { y = ySize - y - 1; }
    if(transpose)
      std::swap(x,y);
    return Location::getLoc(x,y,transpose ? ySize : xSize);
  }
  return getSymLoc(x,y,0,xSize,ySize,1,symmetry);
}

Loc SymmetryHelpers::getSymLoc(int x, int y, const Board& board, int symmetry) {
  return getSymLoc(x,y,board.x_size,board.y_size,symmetry);
}

Loc SymmetryHelpers::getSymLoc(int x, int y, int z, const Board& board, int symmetry) {
  return getSymLoc(x,y,z,board.x_size,board.y_size,board.z_size,symmetry);
}

Loc SymmetryHelpers::getSymLoc(Loc loc, const Board& board, int symmetry) {
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
    return loc;
  return getSymLoc(
    Location::getX(loc,board.x_size),
    Location::getY(loc,board.x_size,board.y_size),
    Location::getZ(loc,board.x_size,board.y_size),
    board,
    symmetry
  );
}

Loc SymmetryHelpers::getSymLoc(Loc loc, int xSize, int ySize, int symmetry) {
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
    return loc;
  return getSymLoc(Location::getX(loc,xSize), Location::getY(loc,xSize), xSize, ySize, symmetry);
}

Loc SymmetryHelpers::getSymLoc(int x, int y, int z, int xSize, int ySize, int zSize, int symmetry) {
  int symX, symY, symZ;
  apply3DSymmetry(x,y,z,xSize,ySize,zSize,symmetry,symX,symY,symZ);
  return Location::getLoc(symX,symY,symZ,xSize,ySize);
}

Loc SymmetryHelpers::getSymLoc(Loc loc, int xSize, int ySize, int zSize, int symmetry) {
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
    return loc;
  return getSymLoc(
    Location::getX(loc,xSize),
    Location::getY(loc,xSize,ySize),
    Location::getZ(loc,xSize,ySize),
    xSize,ySize,zSize,symmetry
  );
}


Board SymmetryHelpers::getSymBoard(const Board& board, int symmetry) {
  Board symBoard(board.x_size, board.y_size, board.z_size);
  for(int z = 0; z<board.z_size; z++) {
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,z,board.x_size,board.y_size);
        Loc symLoc = getSymLoc(x,y,z,board,symmetry);
        bool suc = symBoard.setStone(symLoc,board.colors[loc]);
        assert(suc);
        (void)suc;
      }
    }
  }
  return symBoard;
}

void SymmetryHelpers::markDuplicateMoveLocs(
  const Board& board,
  const BoardHistory& hist,
  const std::vector<int>* onlySymmetries,
  const std::vector<int>& avoidMoves,
  bool* isSymDupLoc,
  std::vector<int>& validSymmetries
) {
  std::fill(isSymDupLoc, isSymDupLoc + Board::MAX_ARR_SIZE, false);
  validSymmetries.clear();
  validSymmetries.reserve(SymmetryHelpers::NUM_SYMMETRIES);
  validSymmetries.push_back(0);


  int symmetrySearchUpperBound = board.isCubical() ? SymmetryHelpers::NUM_SYMMETRIES : SymmetryHelpers::NUM_SYMMETRIES_WITHOUT_TRANSPOSE;

  for(int symmetry = 1; symmetry < symmetrySearchUpperBound; symmetry++) {
    if(onlySymmetries != NULL && !contains(*onlySymmetries,symmetry))
      continue;

    bool isBoardSym = true;

    //check chosen pieces first
    for(int i = 0; i < board.stage; i++) {
      Loc loc = board.midLocs[i];
      if(board.isOnBoard(loc)) {
        Loc symLoc = getSymLoc(loc, board, symmetry);
        if(symLoc != loc)
          isBoardSym = false;
      }
    }

    for(int z = 0; z < board.z_size; z++) {
      for(int y = 0; y < board.y_size; y++) {
        for(int x = 0; x < board.x_size; x++) {
          Loc loc = Location::getLoc(x, y, z, board.x_size, board.y_size);
          Loc symLoc = getSymLoc(x, y, z, board,symmetry);
          bool isStoneSym = (board.colors[loc] == board.colors[symLoc]);
          if(!isStoneSym ) {
            isBoardSym = false;
            break;
          }
        }
        if(!isBoardSym)
          break;
      }
      if(!isBoardSym)
        break;
    }
    if(isBoardSym)
      validSymmetries.push_back(symmetry);
  }

  //The way we iterate is to achieve https://senseis.xmp.net/?PlayingTheFirstMoveInTheUpperRightCorner%2FDiscussion
  //Reverse the iteration order for white, so that natural openings result in white on the left and black on the right
  //as is common now in SGFs
  if(hist.presumedNextMovePla == P_BLACK) {
    for(int z = board.z_size-1; z >= 0; z--) {
      for(int x = board.x_size-1; x >= 0; x--) {
        for(int y = 0; y < board.y_size; y++) {
          Loc loc = Location::getLoc(x, y, z, board.x_size, board.y_size);
          if(avoidMoves.size() > 0 && avoidMoves[loc] > 0)
            continue;
          for(int symmetry: validSymmetries) {
            if(symmetry == 0)
              continue;
            Loc symLoc = getSymLoc(x, y, z, board, symmetry);
            if(!isSymDupLoc[loc] && loc != symLoc)
              isSymDupLoc[symLoc] = true;
          }
        }
      }
    }
  }
  else {
    for(int z = 0; z < board.z_size; z++) {
      for(int x = 0; x < board.x_size; x++) {
        for(int y = board.y_size-1; y >= 0; y--) {
          Loc loc = Location::getLoc(x, y, z, board.x_size, board.y_size);
          if(avoidMoves.size() > 0 && avoidMoves[loc] > 0)
            continue;
          for(int symmetry: validSymmetries) {
            if(symmetry == 0)
              continue;
            Loc symLoc = getSymLoc(x, y, z, board, symmetry);
            if(!isSymDupLoc[loc] && loc != symLoc)
              isSymDupLoc[symLoc] = true;
          }
        }
      }
    }
  }
}

//-------------------------------------------------------------------------------------------------------------

static void setRowBin(float* rowBin, int pos, int feature, float value, int posStride, int featureStride) {
  rowBin[pos * posStride + feature * featureStride] = value;
}

//Currently does NOT depend on history (except for marking ko-illegal spots)
Hash128 NNInputs::getHash(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams
) {
  Hash128 hash =
    BoardHistory::getSituationRulesHash(board, hist, nextPlayer);

  //Fold in whether the game is over or not, since this affects how we compute input features
  //but is not a function necessarily of previous hashed values.
  //If the history is in a weird prolonged state, also treat it similarly.
  if(hist.isGameFinished )
    hash ^= Board::ZOBRIST_GAME_IS_OVER;

  //Fold in asymmetric playout indicator
  if(nnInputParams.playoutDoublingAdvantage != 0) {
    int64_t playoutDoublingsDiscretized = (int64_t)(nnInputParams.playoutDoublingAdvantage*256.0f);
    hash.hash0 += Hash::splitMix64((uint64_t)playoutDoublingsDiscretized);
    hash.hash1 += Hash::basicLCong((uint64_t)playoutDoublingsDiscretized);
    hash ^= MiscNNInputParams::ZOBRIST_PLAYOUT_DOUBLINGS;
  }

  //Fold in policy temperature
  if(nnInputParams.nnPolicyTemperature != 1.0f) {
    int64_t nnPolicyTemperatureDiscretized = (int64_t)(nnInputParams.nnPolicyTemperature*2048.0f);
    hash.hash0 ^= Hash::basicLCong2((uint64_t)nnPolicyTemperatureDiscretized);
    hash.hash1 = Hash::splitMix64(hash.hash1 + (uint64_t)nnPolicyTemperatureDiscretized);
    hash.hash0 += hash.hash1;
    hash ^= MiscNNInputParams::ZOBRIST_NN_POLICY_TEMP;
  }

  // Fold in noResultUtilityForWhite
  int64_t noResultUtilityForWhiteDiscretized = (int64_t)(nnInputParams.noResultUtilityForWhite * 2048.0f);
  hash.hash0 ^= Hash::murmurMix((uint64_t)noResultUtilityForWhiteDiscretized);
  hash.hash1 = Hash::rrmxmx(hash.hash1 + (uint64_t)noResultUtilityForWhiteDiscretized);
  hash.hash0 += hash.hash1;

  return hash;
}

//===========================================================================================
//INPUTSVERSION 7
//===========================================================================================


void NNInputs::fillRowV7(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
) {
  int nnLen = nnXLen * nnYLen;
  assert(nnLen <= NNPos::MAX_NN_LEN);
  assert(board.boardVolume() <= nnLen);
  std::fill(rowBin,rowBin+NUM_FEATURES_SPATIAL_V7*nnLen,false);
  std::fill(rowGlobal,rowGlobal+NUM_FEATURES_GLOBAL_V7,0.0f);

  Player pla = nextPlayer;
  Player opp = getOpp(pla);

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NNInputs::NUM_FEATURES_SPATIAL_V7;
  }
  else {
    featureStride = nnLen;
    posStride = 1;
  }

  GameLogic::ResultsBeforeNN resultsBeforeNN = nnInputParams.resultsBeforeNN;
  if(!resultsBeforeNN.inited) {
    resultsBeforeNN.init(board, hist, nextPlayer);
  }

  for(int loc = 0; loc < board.boardVolume(); loc++) {
    int pos = NNPos::locToPos((Loc)loc, nnLen);
    setRowBin(rowBin,pos,0, 1.0f, posStride, featureStride);

    Color stone = board.colors[loc];
    if(stone == pla)
      setRowBin(rowBin, pos, 1, 1.0f, posStride, featureStride);
    else if(stone == opp)
      setRowBin(rowBin, pos, 2, 1.0f, posStride, featureStride);
    else if(stone == C_BAN)
      setRowBin(rowBin, pos, 3, 1.0f, posStride, featureStride);
  }

  // mid state
  if(board.stage == 0)  // choose
  {
    if(!GameLogic::hasLegalMoveAssumeStage0(board))
      rowGlobal[1] = 1.0f;

  } else if(board.stage == 1)  // place
  {
    rowGlobal[0] = 1.0f;
    Loc chosenMove = board.midLocs[0];
    if(!board.isOnBoard(chosenMove)) {
      std::cout << "nninput: chosen move not on board ";
    } else {
      int pos = NNPos::locToPos(chosenMove, nnLen);
      setRowBin(rowBin, pos, 4, 1.0f, posStride, featureStride);
    }
  } else
    ASSERT_UNREACHABLE;


  //Scoring
  if(hist.rules.loopPassRule == Rules::LOOPDRAW_PASSSCORING) {
  } else if(hist.rules.loopPassRule == Rules::LOOPDRAW_PASSCONTINUE) {
    rowGlobal[2] = 1.0f;
  } else if(hist.rules.loopPassRule == Rules::LOOPLOSE_PASSSCORING) {
    rowGlobal[3] = 1.0f;
  } else if(hist.rules.loopPassRule == Rules::LOOPSCORING_PASSSCORING) {
    rowGlobal[4] = 1.0f;
  } else
    ASSERT_UNREACHABLE;

  float selfKomi = pla == C_BLACK ? hist.rules.komi : -hist.rules.komi;
  rowGlobal[5] = tanh(selfKomi);
  rowGlobal[6] = tanh(selfKomi * 0.3);
  rowGlobal[7] = tanh(selfKomi * 0.1);
  rowGlobal[8] = selfKomi / board.boardArea();

  rowGlobal[9] = (hist.rules.komi + board.boardArea()) % 2;
  
  // Parameter 15 is used because there's actually a discontinuity in how training behavior works when this is
  // nonzero, no matter how slightly.
  if(nnInputParams.playoutDoublingAdvantage != 0) {
    rowGlobal[15] = 1.0;
    rowGlobal[16] = (float)(0.5 * nnInputParams.playoutDoublingAdvantage);
  }

  // noResultUtilityForWhite
  rowGlobal[17] = pla == C_WHITE ? nnInputParams.noResultUtilityForWhite : -nnInputParams.noResultUtilityForWhite;
}
