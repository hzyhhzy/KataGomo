#include "../neuralnet/desc.h"

#include <cmath>
#include <fstream>
#include <zlib.h>

#include "../core/global.h"
#include "../core/fileutils.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninterface.h"

using namespace std;

#if !defined(BYTE_ORDER) || (BYTE_ORDER != LITTLE_ENDIAN && BYTE_ORDER != BIG_ENDIAN)
#error Define BYTE_ORDER to be equal to either LITTLE_ENDIAN or BIG_ENDIAN
#endif

static void checkWeightFinite(float f, const string& name) {
  if(!isfinite(f))
    throw StringError(name + ": Nan or infinite neural net weight or parameter");
}
#define CHECKFINITE(x, name) \
  { checkWeightFinite((x), name); }

//For some strange reason, this function is noticeably faster than
//float x; in >> x;
static float readFloatFast(istream& in, string& tmp) {
  in >> tmp;
  char* endPtr;
  const char* cstr = tmp.c_str();
  float x = strtof(cstr,&endPtr);
  if(endPtr == cstr)
    in.setstate(ios_base::failbit);
  return x;
}

static void readFloats(istream& in, size_t numFloats, bool binaryFloats, const string& name, vector<float>& buf) {
  buf.resize(numFloats);
  if(!binaryFloats) {
    string tmp;
    for(size_t i = 0; i<numFloats; i++) {
      float x = readFloatFast(in,tmp);
      CHECKFINITE(x,name);
      buf[i] = x;
    }
    if(in.fail())
      throw StringError(name + ": could not read float weights. Invalid model - perhaps you are trying to load a .bin.gz model as a .txt.gz model?");
  }
  else {
    //KataGo hacky model format - "@BIN@" followed by the expected number of 32 bit floats, in little-endian binary
    assert(sizeof(float) == 4);
    {
      string s;
      int numCharsBeforeAt = 0;
      while((char)in.get() != '@') {
        numCharsBeforeAt++;
        //Something is wrong, there should not be this much whitespace
        if(numCharsBeforeAt > 100 || in.fail()) {
          throw StringError(name + ": could not read float weights. Invalid model - perhaps you are trying to load a .txt.gz model as a .bin.gz model?");
        }
      }
      s += (char)in.get();
      s += (char)in.get();
      s += (char)in.get();
      s += (char)in.get();
      if(s != "BIN@")
        throw StringError(name + ": did not find expected header for binary float block");
    }
    float* data = buf.data();
    char* bytes = (char*)data;
    in.read(bytes, numFloats*sizeof(float));

    if(in.fail())
      throw StringError(name + ": did not find the expected number of floats in binary float block");

#if BYTE_ORDER == BIG_ENDIAN
    for(size_t i = 0; i<numFloats; i++) {
      //Reverse byte order for big endian
      std::swap(bytes[i*4 + 0], bytes[i*4 + 3]);
      std::swap(bytes[i*4 + 1], bytes[i*4 + 2]);
    }
#endif
    for(size_t i = 0; i<numFloats; i++) {
      CHECKFINITE(buf[i],name);
    }
  }
}

//-----------------------------------------------------------------------------

static void parseResidualBlockStack(
  std::istream& in,
  int version,
  bool binaryFloats,
  std::string name,
  int numBlocks,
  int trunkNumChannels,
  std::vector<std::pair<int, unique_ptr_void>>& blocks
);


//-----------------------------------------------------------------------------

ConvLayerDesc::ConvLayerDesc()
  : convYSize(0), convXSize(0), inChannels(0), outChannels(0), dilationY(1), dilationX(1) {}

ConvLayerDesc::ConvLayerDesc(istream& in, bool binaryFloats) {
  in >> name;
  in >> convYSize;
  in >> convXSize;
  in >> inChannels;
  in >> outChannels;
  in >> dilationY;
  in >> dilationX;

  if(in.fail())
    throw StringError(name + ": convlayer failed to parse sizes and channels and dilations");

  if(convXSize <= 0 || convYSize <= 0)
    throw StringError(name + ": convolution filter sizes must be positive");
  if(inChannels <= 0 || outChannels <= 0)
    throw StringError(name + ": number of in and out channels must be positive");
  if(dilationX <= 0 || dilationY <= 0)
    throw StringError(name + ": dilation factors must be positive");
  if(convXSize % 2 != 1 || convYSize % 2 != 1)
    throw StringError(name + ": convolution filter sizes must be odd, found even sizes");

  // Model file order is y,x,ic,oc
  // Cuda's order is oc,ic,y,x
  int numWeights = convYSize * convXSize * inChannels * outChannels;
  weights.resize(numWeights);
  int ocStride = convYSize * convXSize * inChannels;
  int icStride = convYSize * convXSize;
  int yStride = convXSize;
  int xStride = 1;

  vector<float> floats;
  readFloats(in, (size_t)convYSize * convXSize * inChannels * outChannels, binaryFloats, name, floats);
  size_t idx = 0;
  for(int y = 0; y < convYSize; y++) {
    for(int x = 0; x < convXSize; x++) {
      for(int ic = 0; ic < inChannels; ic++) {
        for(int oc = 0; oc < outChannels; oc++) {
          float w = floats[idx++];
          weights[oc * ocStride + ic * icStride + y * yStride + x * xStride] = w;
        }
      }
    }
  }
  if(in.fail())
    throw StringError(name + ": convlayer failed to expected number of float weights");
}

ConvLayerDesc::ConvLayerDesc(ConvLayerDesc&& other) {
  *this = std::move(other);
}

ConvLayerDesc& ConvLayerDesc::operator=(ConvLayerDesc&& other) {
  name = std::move(other.name);
  convYSize = other.convYSize;
  convXSize = other.convXSize;
  inChannels = other.inChannels;
  outChannels = other.outChannels;
  dilationY = other.dilationY;
  dilationX = other.dilationX;
  weights = std::move(other.weights);
  return *this;
}

//-----------------------------------------------------------------------------

BatchNormLayerDesc::BatchNormLayerDesc() : numChannels(0), epsilon(0.001f), hasScale(false), hasBias(false) {}

BatchNormLayerDesc::BatchNormLayerDesc(istream& in, bool binaryFloats) {
  in >> name;
  in >> numChannels;
  in >> epsilon;
  in >> hasScale;
  in >> hasBias;

  if(in.fail())
    throw StringError(name + ": bnlayer failed to parse num channels and epsilon and hasScale and hasBias");

  if(numChannels < 1)
    throw StringError(name + ": numChannels (" + Global::intToString(numChannels) + ") < 1");
  if(epsilon <= 0)
    throw StringError(name + ": epsilon (" + Global::floatToString(epsilon) + ") <= 0");

  vector<float> floats;
  readFloats(in, (size_t)numChannels, binaryFloats, name, floats);
  mean = floats;
  readFloats(in, (size_t)numChannels, binaryFloats, name, floats);
  variance = floats;

  if(hasScale) {
    readFloats(in, (size_t)numChannels, binaryFloats, name, floats);
    scale = floats;
  }
  else {
    scale.resize(numChannels);
    for(int c = 0; c < numChannels; c++)
      scale[c] = 1.0;
  }

  if(hasBias) {
    readFloats(in, (size_t)numChannels, binaryFloats, name, floats);
    bias = floats;
  }
  else {
    bias.resize(numChannels);
    for(int c = 0; c < numChannels; c++)
      bias[c] = 1.0;
  }

  if(in.fail())
    throw StringError(
      name + ": bnlayer failed to parse expected number of batch norm mean, variance, bias, scale values");
}

BatchNormLayerDesc::BatchNormLayerDesc(BatchNormLayerDesc&& other) {
  *this = std::move(other);
}

BatchNormLayerDesc& BatchNormLayerDesc::operator=(BatchNormLayerDesc&& other) {
  name = std::move(other.name);
  numChannels = other.numChannels;
  epsilon = other.epsilon;
  hasScale = other.hasScale;
  hasBias = other.hasBias;
  mean = std::move(other.mean);
  variance = std::move(other.variance);
  scale = std::move(other.scale);
  bias = std::move(other.bias);
  return *this;
}

//-----------------------------------------------------------------------------

ActivationLayerDesc::ActivationLayerDesc() : activation(ACTIVATION_RELU) {}

ActivationLayerDesc::ActivationLayerDesc(istream& in, int version) {
  in >> name;
  if(version >= 11 && version != 101) {
    string kind;
    in >> kind;
    if(kind == "ACTIVATION_IDENTITY")
      activation = ACTIVATION_IDENTITY;
    else if(kind == "ACTIVATION_RELU")
      activation = ACTIVATION_RELU;
    else if(kind == "ACTIVATION_MISH")
      activation = ACTIVATION_MISH;
    else if(kind == "ACTIVATION_SILU")
      activation = ACTIVATION_SILU;
    else
      throw StringError(
        name + ": unknown activation " + kind
      );
  }
  else {
    activation = ACTIVATION_RELU;
  }
}

ActivationLayerDesc::ActivationLayerDesc(ActivationLayerDesc&& other) {
  *this = std::move(other);
}

ActivationLayerDesc& ActivationLayerDesc::operator=(ActivationLayerDesc&& other) {
  name = std::move(other.name);
  activation = other.activation;
  return *this;
}

//-----------------------------------------------------------------------------

MatMulLayerDesc::MatMulLayerDesc() : inChannels(0), outChannels(0) {}

MatMulLayerDesc::MatMulLayerDesc(istream& in, bool binaryFloats) {
  in >> name;
  in >> inChannels;
  in >> outChannels;

  if(in.fail())
    throw StringError(name + ": matmullayer failed to parse num channels");
  if(inChannels <= 0 || outChannels <= 0)
    throw StringError(name + ": number of in and out channels must be positive");

  // Model file order is ic,oc
  // Cublas order used is also ic,oc since we transpose
  int numWeights = inChannels * outChannels;
  weights.resize(numWeights);
  int icStride = outChannels;
  int ocStride = 1;

  vector<float> floats;
  readFloats(in, (size_t)inChannels * outChannels, binaryFloats, name, floats);
  size_t idx = 0;
  for(int ic = 0; ic < inChannels; ic++) {
    for(int oc = 0; oc < outChannels; oc++) {
      float w = floats[idx++];
      weights[oc * ocStride + ic * icStride] = w;
    }
  }
  if(in.fail())
    throw StringError(name + ": matmullayer failed to parse expected number of matmul weights");
}

MatMulLayerDesc::MatMulLayerDesc(MatMulLayerDesc&& other) {
  *this = std::move(other);
}

MatMulLayerDesc& MatMulLayerDesc::operator=(MatMulLayerDesc&& other) {
  name = std::move(other.name);
  inChannels = other.inChannels;
  outChannels = other.outChannels;
  weights = std::move(other.weights);
  return *this;
}

//-----------------------------------------------------------------------------

MatBiasLayerDesc::MatBiasLayerDesc() : numChannels(0) {}

MatBiasLayerDesc::MatBiasLayerDesc(istream& in, bool binaryFloats) {
  in >> name;
  in >> numChannels;

  if(in.fail())
    throw StringError(name + ": matbiaslayer failed to parse num channels");
  if(numChannels <= 0)
    throw StringError(name + ": number of channels must be positive");

  weights.resize(numChannels);

  vector<float> floats;
  readFloats(in, (size_t)numChannels, binaryFloats, name, floats);
  weights = floats;

  if(in.fail())
    throw StringError(name + ": matbiaslayer failed to parse expected number of matbias weights");
}

MatBiasLayerDesc::MatBiasLayerDesc(MatBiasLayerDesc&& other) {
  *this = std::move(other);
}

MatBiasLayerDesc& MatBiasLayerDesc::operator=(MatBiasLayerDesc&& other) {
  name = std::move(other.name);
  numChannels = other.numChannels;
  weights = std::move(other.weights);
  return *this;
}

//-----------------------------------------------------------------------------

TransformerRMSNormDesc::TransformerRMSNormDesc()
  : numChannels(0), epsilon(0.0f) {}

TransformerRMSNormDesc::TransformerRMSNormDesc(istream& in, bool binaryFloats) {
  in >> name;
  in >> numChannels;
  in >> epsilon;
  if(in.fail())
    throw StringError(name + ": transformer rmsnorm failed to parse parameters");
  if(numChannels < 1)
    throw StringError(name + ": transformer rmsnorm numChannels must be positive");
  if(!isfinite(epsilon) || epsilon <= 0.0f || epsilon > 1.0f)
    throw StringError(name + ": transformer rmsnorm epsilon must be positive and at most 1");
  readFloats(in, (size_t)numChannels, binaryFloats, name, weight);
  if(in.fail())
    throw StringError(name + ": transformer rmsnorm failed to parse weights");
}

TransformerRMSNormDesc::TransformerRMSNormDesc(TransformerRMSNormDesc&& other) {
  *this = std::move(other);
}

TransformerRMSNormDesc& TransformerRMSNormDesc::operator=(TransformerRMSNormDesc&& other) {
  name = std::move(other.name);
  numChannels = other.numChannels;
  epsilon = other.epsilon;
  weight = std::move(other.weight);
  return *this;
}

//-----------------------------------------------------------------------------

TransformerAttentionDesc::TransformerAttentionDesc()
  : numHeads(0), numKVHeads(0), qHeadDim(0), vHeadDim(0),
    useRope(false), learnableRope(false),
    ropeNumKVHeads(0), ropeNumPairs(0), ropeTheta(0.0f) {}

TransformerAttentionDesc::TransformerAttentionDesc(istream& in, bool binaryFloats) {
  in >> name;
  in >> numHeads;
  in >> numKVHeads;
  in >> qHeadDim;
  in >> vHeadDim;
  int useRopeInt;
  int learnableRopeInt;
  in >> useRopeInt;
  in >> learnableRopeInt;
  if((useRopeInt != 0 && useRopeInt != 1) ||
     (learnableRopeInt != 0 && learnableRopeInt != 1))
    throw StringError(name + ": transformer attention rope flags must be 0 or 1");
  useRope = useRopeInt != 0;
  learnableRope = learnableRopeInt != 0;

  if(in.fail())
    throw StringError(name + ": transformer attention block failed to parse header");
  if(numHeads < 1 || numKVHeads < 1 || numHeads % numKVHeads != 0)
    throw StringError(name + ": transformer attention head counts are invalid");
  if(qHeadDim < 1 || vHeadDim < 1 || qHeadDim % 2 != 0)
    throw StringError(name + ": transformer attention head dimensions are invalid");
  if(learnableRope && !useRope)
    throw StringError(name + ": learnableRope requires useRope");

  preLN = TransformerRMSNormDesc(in, binaryFloats);
  qProj = MatMulLayerDesc(in, binaryFloats);
  kProj = MatMulLayerDesc(in, binaryFloats);
  vProj = MatMulLayerDesc(in, binaryFloats);
  outProj = MatMulLayerDesc(in, binaryFloats);

  if(qProj.inChannels != preLN.numChannels ||
     kProj.inChannels != preLN.numChannels ||
     vProj.inChannels != preLN.numChannels)
    throw StringError(name + ": q/k/v projection input channels do not match preLN");
  if(qProj.outChannels != numHeads * qHeadDim)
    throw StringError(name + ": q projection output channels do not match attention geometry");
  if(kProj.outChannels != numKVHeads * qHeadDim)
    throw StringError(name + ": k projection output channels do not match attention geometry");
  if(vProj.outChannels != numKVHeads * vHeadDim)
    throw StringError(name + ": v projection output channels do not match attention geometry");
  if(outProj.inChannels != numHeads * vHeadDim || outProj.outChannels != preLN.numChannels)
    throw StringError(name + ": output projection channels do not match attention geometry");

  ropeNumKVHeads = 0;
  ropeNumPairs = 0;
  ropeTheta = 0.0f;
  if(useRope) {
    if(learnableRope) {
      string ropeFreqsName;
      int ropeDim2;
      in >> ropeFreqsName;
      in >> ropeNumKVHeads;
      in >> ropeNumPairs;
      in >> ropeDim2;
      if(in.fail())
        throw StringError(name + ": failed to parse learnable rope header");
      if(ropeNumKVHeads != numKVHeads || ropeNumPairs != qHeadDim / 2 || ropeDim2 != 2)
        throw StringError(name + ": learnable rope shape does not match attention geometry");
      readFloats(
        in,
        (size_t)ropeNumKVHeads * (size_t)ropeNumPairs * 2,
        binaryFloats,
        ropeFreqsName,
        ropeFreqs
      );
    }
    else {
      string ropeThetaName;
      in >> ropeThetaName;
      in >> ropeTheta;
      if(in.fail() || !isfinite(ropeTheta) || ropeTheta <= 0.0f)
        throw StringError(name + ": fixed rope theta must be positive");
    }
  }
  if(in.fail())
    throw StringError(name + ": transformer attention block parse failure");
}

TransformerAttentionDesc::TransformerAttentionDesc(TransformerAttentionDesc&& other) {
  *this = std::move(other);
}

TransformerAttentionDesc& TransformerAttentionDesc::operator=(TransformerAttentionDesc&& other) {
  name = std::move(other.name);
  numHeads = other.numHeads;
  numKVHeads = other.numKVHeads;
  qHeadDim = other.qHeadDim;
  vHeadDim = other.vHeadDim;
  useRope = other.useRope;
  learnableRope = other.learnableRope;
  preLN = std::move(other.preLN);
  qProj = std::move(other.qProj);
  kProj = std::move(other.kProj);
  vProj = std::move(other.vProj);
  outProj = std::move(other.outProj);
  ropeNumKVHeads = other.ropeNumKVHeads;
  ropeNumPairs = other.ropeNumPairs;
  ropeFreqs = std::move(other.ropeFreqs);
  ropeTheta = other.ropeTheta;
  return *this;
}

void TransformerAttentionDesc::computeRopeCosSin(
  int nnXLen,
  int nnYLen,
  int paddedNNXYLen,
  vector<float>& cosTable,
  vector<float>& sinTable
) const {
  if(!useRope)
    throw StringError("computeRopeCosSin called when useRope is false");
  const int nnXYLen = nnXLen * nnYLen;
  if(nnXLen < 1 || nnYLen < 1 || paddedNNXYLen < nnXYLen)
    throw StringError("computeRopeCosSin received invalid board dimensions");

  const int numPairs = qHeadDim / 2;
  if(learnableRope) {
    if(ropeNumKVHeads != numKVHeads ||
       ropeNumPairs != numPairs ||
       ropeFreqs.size() != (size_t)numKVHeads * (size_t)numPairs * 2)
      throw StringError("learnable rope descriptor is internally inconsistent");
    cosTable.assign((size_t)numKVHeads * numPairs * paddedNNXYLen, 0.0f);
    sinTable.assign((size_t)numKVHeads * numPairs * paddedNNXYLen, 0.0f);
    for(int h = 0; h < numKVHeads; h++) {
      for(int p = 0; p < numPairs; p++) {
        const float freqX = ropeFreqs[((size_t)h * numPairs + p) * 2];
        const float freqY = ropeFreqs[((size_t)h * numPairs + p) * 2 + 1];
        for(int y = 0; y < nnYLen; y++) {
          for(int x = 0; x < nnXLen; x++) {
            const int xy = y * nnXLen + x;
            const float angle = (float)x * freqX + (float)y * freqY;
            const size_t idx = ((size_t)h * numPairs + p) * paddedNNXYLen + xy;
            cosTable[idx] = cosf(angle);
            sinTable[idx] = sinf(angle);
          }
        }
      }
    }
  }
  else {
    if(numPairs % 2 != 0)
      throw StringError("fixed 2D rope requires an even number of coordinate pairs");
    const int pairsPerDim = numPairs / 2;
    const int dimHalf = qHeadDim / 2;
    cosTable.assign((size_t)numPairs * paddedNNXYLen, 0.0f);
    sinTable.assign((size_t)numPairs * paddedNNXYLen, 0.0f);
    for(int p = 0; p < numPairs; p++) {
      const int coordinatePair = p < pairsPerDim ? p : p - pairsPerDim;
      const float freq = 1.0f / powf(ropeTheta, (float)(2 * coordinatePair) / (float)dimHalf);
      for(int y = 0; y < nnYLen; y++) {
        for(int x = 0; x < nnXLen; x++) {
          const int xy = y * nnXLen + x;
          const float angle = (float)(p < pairsPerDim ? y : x) * freq;
          const size_t idx = (size_t)p * paddedNNXYLen + xy;
          cosTable[idx] = cosf(angle);
          sinTable[idx] = sinf(angle);
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------

TransformerFFNDesc::TransformerFFNDesc()
  : numChannels(0), ffnChannels(0), useSwiGLU(false) {}

TransformerFFNDesc::TransformerFFNDesc(istream& in, bool binaryFloats) {
  in >> name;
  in >> numChannels;
  in >> ffnChannels;
  int useSwiGLUInt;
  in >> useSwiGLUInt;
  if(useSwiGLUInt != 0 && useSwiGLUInt != 1)
    throw StringError(name + ": transformer ffn useSwiGLU flag must be 0 or 1");
  useSwiGLU = useSwiGLUInt != 0;
  if(in.fail())
    throw StringError(name + ": transformer ffn block failed to parse header");
  if(numChannels < 1 || ffnChannels < 1)
    throw StringError(name + ": transformer ffn channel counts must be positive");

  preLN = TransformerRMSNormDesc(in, binaryFloats);
  linear1 = MatMulLayerDesc(in, binaryFloats);
  if(useSwiGLU)
    linearGate = MatMulLayerDesc(in, binaryFloats);
  linear2 = MatMulLayerDesc(in, binaryFloats);

  if(preLN.numChannels != numChannels)
    throw StringError(name + ": ffn preLN channels do not match trunk channels");
  if(linear1.inChannels != numChannels || linear1.outChannels != ffnChannels)
    throw StringError(name + ": ffn linear1 channels do not match header");
  if(useSwiGLU &&
     (linearGate.inChannels != numChannels || linearGate.outChannels != ffnChannels))
    throw StringError(name + ": ffn gate channels do not match header");
  if(linear2.inChannels != ffnChannels || linear2.outChannels != numChannels)
    throw StringError(name + ": ffn linear2 channels do not match header");
  if(in.fail())
    throw StringError(name + ": transformer ffn block parse failure");
}

TransformerFFNDesc::TransformerFFNDesc(TransformerFFNDesc&& other) {
  *this = std::move(other);
}

TransformerFFNDesc& TransformerFFNDesc::operator=(TransformerFFNDesc&& other) {
  name = std::move(other.name);
  numChannels = other.numChannels;
  ffnChannels = other.ffnChannels;
  useSwiGLU = other.useSwiGLU;
  preLN = std::move(other.preLN);
  linear1 = std::move(other.linear1);
  linearGate = std::move(other.linearGate);
  linear2 = std::move(other.linear2);
  return *this;
}

//-----------------------------------------------------------------------------

ResidualBlockDesc::ResidualBlockDesc() {}

ResidualBlockDesc::ResidualBlockDesc(istream& in, int version, bool binaryFloats) {
  in >> name;
  if(in.fail())
    throw StringError(name + ": res block failed to parse name");

  preBN = BatchNormLayerDesc(in,binaryFloats);
  preActivation = ActivationLayerDesc(in,version);
  regularConv = ConvLayerDesc(in,binaryFloats);
  midBN = BatchNormLayerDesc(in,binaryFloats);
  midActivation = ActivationLayerDesc(in,version);
  finalConv = ConvLayerDesc(in,binaryFloats);

  if(preBN.numChannels != regularConv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": preBN.numChannels (%d) != regularConv.inChannels (%d)", preBN.numChannels, regularConv.inChannels));
  if(midBN.numChannels != regularConv.outChannels)
    throw StringError(
      name + Global::strprintf(
               ": midBN.numChannels (%d) != regularConv.outChannels (%d)", midBN.numChannels, regularConv.outChannels));
  if(midBN.numChannels != finalConv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": midBN.numChannels (%d) != finalConv.inChannels (%d)", midBN.numChannels, finalConv.inChannels));

  if(in.fail())
    throw StringError(name + ": res block parse failure (istream fail() return true)");
}

ResidualBlockDesc::ResidualBlockDesc(ResidualBlockDesc&& other) {
  *this = std::move(other);
}

ResidualBlockDesc& ResidualBlockDesc::operator=(ResidualBlockDesc&& other) {
  name = std::move(other.name);
  preBN = std::move(other.preBN);
  preActivation = std::move(other.preActivation);
  regularConv = std::move(other.regularConv);
  midBN = std::move(other.midBN);
  midActivation = std::move(other.midActivation);
  finalConv = std::move(other.finalConv);
  return *this;
}

void ResidualBlockDesc::iterConvLayers(std::function<void(const ConvLayerDesc& desc)> f) const {
  f(regularConv);
  f(finalConv);
}


//-----------------------------------------------------------------------------

GlobalPoolingResidualBlockDesc::GlobalPoolingResidualBlockDesc() {}

GlobalPoolingResidualBlockDesc::GlobalPoolingResidualBlockDesc(istream& in, int vrsn, bool binaryFloats) {
  in >> name;
  if(in.fail())
    throw StringError(name + ": gpool res block failed to parse name");
  version = vrsn;
  preBN = BatchNormLayerDesc(in,binaryFloats);
  preActivation = ActivationLayerDesc(in,version);
  regularConv = ConvLayerDesc(in,binaryFloats);
  gpoolConv = ConvLayerDesc(in,binaryFloats);
  gpoolBN = BatchNormLayerDesc(in,binaryFloats);
  gpoolActivation = ActivationLayerDesc(in,version);
  gpoolToBiasMul = MatMulLayerDesc(in,binaryFloats);
  midBN = BatchNormLayerDesc(in,binaryFloats);
  midActivation = ActivationLayerDesc(in,version);
  finalConv = ConvLayerDesc(in,binaryFloats);

  if(preBN.numChannels != regularConv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": preBN.numChannels (%d) != regularConv.inChannels (%d)", preBN.numChannels, regularConv.inChannels));
  if(preBN.numChannels != gpoolConv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": preBN.numChannels (%d) != gpoolConv.inChannels (%d)", preBN.numChannels, gpoolConv.inChannels));
  if(gpoolBN.numChannels != gpoolConv.outChannels)
    throw StringError(
      name + Global::strprintf(
               ": gpoolBN.numChannels (%d) != gpoolConv.outChannels (%d)", gpoolBN.numChannels, gpoolConv.outChannels));
  if(gpoolBN.numChannels * 3 != gpoolToBiasMul.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": gpoolBN.numChannels * 3 (%d) != gpoolToBiasMul.inChannels (%d)",
               gpoolBN.numChannels * 3,
               gpoolToBiasMul.inChannels));
  if(midBN.numChannels != regularConv.outChannels)
    throw StringError(
      name + Global::strprintf(
               ": midBN.numChannels (%d) != regularConv.outChannels (%d)", midBN.numChannels, regularConv.outChannels));
  if(midBN.numChannels != gpoolToBiasMul.outChannels)
    throw StringError(
      name +
      Global::strprintf(
        ": midBN.numChannels (%d) != gpoolToBiasMul.outChannels (%d)", midBN.numChannels, gpoolToBiasMul.outChannels));
  if(midBN.numChannels != finalConv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": midBN.numChannels (%d) != finalConv.inChannels (%d)", midBN.numChannels, finalConv.inChannels));

  if(in.fail())
    throw StringError(name + ": gpool res block parse failure (istream fail() return true)");
}

GlobalPoolingResidualBlockDesc::GlobalPoolingResidualBlockDesc(GlobalPoolingResidualBlockDesc&& other) {
  *this = std::move(other);
}

GlobalPoolingResidualBlockDesc& GlobalPoolingResidualBlockDesc::operator=(GlobalPoolingResidualBlockDesc&& other) {
  name = std::move(other.name);
  preBN = std::move(other.preBN);
  preActivation = std::move(other.preActivation);
  regularConv = std::move(other.regularConv);
  gpoolConv = std::move(other.gpoolConv);
  gpoolBN = std::move(other.gpoolBN);
  gpoolActivation = std::move(other.gpoolActivation);
  gpoolToBiasMul = std::move(other.gpoolToBiasMul);
  midBN = std::move(other.midBN);
  midActivation = std::move(other.midActivation);
  finalConv = std::move(other.finalConv);
  return *this;
}

void GlobalPoolingResidualBlockDesc::iterConvLayers(std::function<void(const ConvLayerDesc& desc)> f) const {
  f(regularConv);
  f(gpoolConv);
  f(finalConv);
}

//-----------------------------------------------------------------------------

NestedBottleneckResidualBlockDesc::NestedBottleneckResidualBlockDesc() {}

NestedBottleneckResidualBlockDesc::NestedBottleneckResidualBlockDesc(istream& in, int version, bool binaryFloats) {
  in >> name;
  if(in.fail())
    throw StringError(name + ": res block failed to parse name");
  in >> numBlocks;
  if(in.fail())
    throw StringError(name + ": nested bottleneck res block failed to parse num blocks");
  if(numBlocks < 1)
    throw StringError(name + ": nested bottleneck res block num blocks must be positive");

  preBN = BatchNormLayerDesc(in,binaryFloats);
  preActivation = ActivationLayerDesc(in,version);
  preConv = ConvLayerDesc(in,binaryFloats);

  parseResidualBlockStack(in, version, binaryFloats, name, numBlocks, preConv.outChannels, blocks);

  postBN = BatchNormLayerDesc(in,binaryFloats);
  postActivation = ActivationLayerDesc(in,version);
  postConv = ConvLayerDesc(in,binaryFloats);

  if(preBN.numChannels != preConv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": preBN.numChannels (%d) != preConv.inChannels (%d)", preBN.numChannels, preConv.inChannels));
  if(postBN.numChannels != preConv.outChannels)
    throw StringError(
      name + Global::strprintf(
               ": postBN.numChannels (%d) != preConv.outChannels (%d)", postBN.numChannels, preConv.outChannels));
  if(postBN.numChannels != postConv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": postBN.numChannels (%d) != postConv.inChannels (%d)", postBN.numChannels, postConv.inChannels));

  if(in.fail())
    throw StringError(name + ": nested res block parse failure (istream fail() return true)");
}

NestedBottleneckResidualBlockDesc::NestedBottleneckResidualBlockDesc(NestedBottleneckResidualBlockDesc&& other) {
  *this = std::move(other);
}

NestedBottleneckResidualBlockDesc& NestedBottleneckResidualBlockDesc::operator=(NestedBottleneckResidualBlockDesc&& other) {
  name = std::move(other.name);
  numBlocks = other.numBlocks;
  preBN = std::move(other.preBN);
  preActivation = std::move(other.preActivation);
  preConv = std::move(other.preConv);
  blocks = std::move(other.blocks);
  postBN = std::move(other.postBN);
  postActivation = std::move(other.postActivation);
  postConv = std::move(other.postConv);
  return *this;
}

void NestedBottleneckResidualBlockDesc::iterConvLayers(std::function<void(const ConvLayerDesc& desc)> f) const {
  f(preConv);
  for(int i = 0; i < blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlockDesc* desc = (ResidualBlockDesc*)blocks[i].second.get();
      desc->iterConvLayers(f);
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlockDesc* desc = (GlobalPoolingResidualBlockDesc*)blocks[i].second.get();
      desc->iterConvLayers(f);
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlockDesc* desc = (NestedBottleneckResidualBlockDesc*)blocks[i].second.get();
      desc->iterConvLayers(f);
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND ||
            blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      // Transformer projections are matmuls, not convolutions.
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
  f(postConv);
}

//-----------------------------------------------------------------------------

static void parseResidualBlockStack(
  std::istream& in,
  int version,
  bool binaryFloats,
  std::string name,
  int numBlocks,
  int trunkNumChannels,
  std::vector<std::pair<int, unique_ptr_void>>& blocks
) {
  string kind;
  for(int i = 0; i < numBlocks; i++) {
    in >> kind;
    if(in.fail())
      throw StringError(name + ": failed to parse block kind");
    if(kind == "ordinary_block") {
      unique_ptr_void descPtr = make_unique_void(new ResidualBlockDesc(in,version,binaryFloats));
      ResidualBlockDesc& desc = *((ResidualBlockDesc*)descPtr.get());

      if(desc.preBN.numChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s preBN.numChannels (%d) != trunkNumChannels (%d)",
                   desc.name.c_str(),
                   desc.preBN.numChannels,
                   trunkNumChannels));
      if(desc.finalConv.outChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s finalConv.outChannels (%d) != trunkNumChannels (%d)",
                   desc.name.c_str(),
                   desc.finalConv.outChannels,
                   trunkNumChannels));

      blocks.push_back(make_pair(ORDINARY_BLOCK_KIND, std::move(descPtr)));
    }
    else if(kind == "gpool_block") {
      unique_ptr_void descPtr = make_unique_void(new GlobalPoolingResidualBlockDesc(in, version, binaryFloats));
      GlobalPoolingResidualBlockDesc& desc = *((GlobalPoolingResidualBlockDesc*)descPtr.get());

      if(desc.preBN.numChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s preBN.numChannels (%d) != trunkNumChannels (%d)",
                   desc.name.c_str(),
                   desc.preBN.numChannels,
                   trunkNumChannels));
      if(desc.finalConv.outChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s finalConv.outChannels (%d) != trunkNumChannels (%d)",
                   desc.name.c_str(),
                   desc.finalConv.outChannels,
                   trunkNumChannels));

      blocks.push_back(make_pair(GLOBAL_POOLING_BLOCK_KIND, std::move(descPtr)));
    }
    else if(kind == "nested_bottleneck_block") {
      unique_ptr_void descPtr = make_unique_void(new NestedBottleneckResidualBlockDesc(in,version,binaryFloats));
      NestedBottleneckResidualBlockDesc& desc = *((NestedBottleneckResidualBlockDesc*)descPtr.get());

      if(desc.preBN.numChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s preBN.numChannels (%d) != trunkNumChannels (%d)",
                   desc.name.c_str(),
                   desc.preBN.numChannels,
                   trunkNumChannels));
      if(desc.postConv.outChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s postConv.outChannels (%d) != trunkNumChannels (%d)",
                   desc.name.c_str(),
                   desc.postConv.outChannels,
                   trunkNumChannels));

      blocks.push_back(make_pair(NESTED_BOTTLENECK_BLOCK_KIND, std::move(descPtr)));
    }
    else if(kind == "transformer_attention_block") {
      unique_ptr_void descPtr = make_unique_void(new TransformerAttentionDesc(in,binaryFloats));
      TransformerAttentionDesc& desc = *((TransformerAttentionDesc*)descPtr.get());
      if(desc.preLN.numChannels != trunkNumChannels ||
         desc.qProj.inChannels != trunkNumChannels ||
         desc.outProj.outChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s transformer attention channels do not match trunkNumChannels (%d)",
                   desc.name.c_str(),
                   trunkNumChannels));
      blocks.push_back(make_pair(TRANSFORMER_ATTENTION_BLOCK_KIND, std::move(descPtr)));
    }
    else if(kind == "transformer_ffn_block") {
      unique_ptr_void descPtr = make_unique_void(new TransformerFFNDesc(in,binaryFloats));
      TransformerFFNDesc& desc = *((TransformerFFNDesc*)descPtr.get());
      if(desc.numChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s transformer ffn channels (%d) do not match trunkNumChannels (%d)",
                   desc.name.c_str(),
                   desc.numChannels,
                   trunkNumChannels));
      blocks.push_back(make_pair(TRANSFORMER_FFN_BLOCK_KIND, std::move(descPtr)));
    }
    else
      throw StringError(name + ": found unknown block kind: " + kind);

    if(in.fail())
      throw StringError(name + ": trunk istream fail after parsing block");
  }
}

//-----------------------------------------------------------------------------


TrunkDesc::TrunkDesc()
  : version(-1),
    numBlocks(0),
    trunkNumChannels(0),
    midNumChannels(0),
    regularNumChannels(0),
    gpoolNumChannels(0) {}

TrunkDesc::TrunkDesc(istream& in, int vrsn, bool binaryFloats) {
  in >> name;
  version = vrsn;
  in >> numBlocks;
  in >> trunkNumChannels;
  in >> midNumChannels;
  in >> regularNumChannels;
  int dilatedNumChannels; //unused
  in >> dilatedNumChannels;
  in >> gpoolNumChannels;

  if(in.fail())
    throw StringError(name + ": trunk failed to parse num blocks or various channel parameters");
  if(numBlocks < 1)
    throw StringError(name + ": trunk num blocks must be positive");
  if(
    trunkNumChannels <= 0 || midNumChannels <= 0 || regularNumChannels <= 0 ||
    gpoolNumChannels <= 0)
    throw StringError(name + ": all numbers of channels must be positive");

  initialConv = ConvLayerDesc(in,binaryFloats);
  if(initialConv.outChannels != trunkNumChannels)
    throw StringError(
      name + Global::strprintf(
               ": %s initialConv.outChannels (%d) != trunkNumChannels (%d)",
               initialConv.name.c_str(),
               initialConv.outChannels,
               trunkNumChannels));

  initialMatMul = MatMulLayerDesc(in,binaryFloats);
  if(initialMatMul.outChannels != trunkNumChannels)
    throw StringError(
      name + Global::strprintf(
               ": %s initialMatMul.outChannels (%d) != trunkNumChannels (%d)",
               initialMatMul.name.c_str(),
               initialMatMul.outChannels,
               trunkNumChannels));

  parseResidualBlockStack(in, version, binaryFloats, name, numBlocks, trunkNumChannels, blocks);

  trunkTipBN = BatchNormLayerDesc(in,binaryFloats);
  trunkTipActivation = ActivationLayerDesc(in,version);

  if(trunkTipBN.numChannels != trunkNumChannels)
    throw StringError(
      name + Global::strprintf(
               ": trunkTipBN.numChannels (%d) != trunkNumChannels (%d)", trunkTipBN.numChannels, trunkNumChannels));

  if(in.fail())
    throw StringError(name + ": trunk istream fail after parsing tip");
}

TrunkDesc::~TrunkDesc() {
}

TrunkDesc::TrunkDesc(TrunkDesc&& other) {
  name = std::move(other.name);
  version = other.version;
  numBlocks = other.numBlocks;
  trunkNumChannels = other.trunkNumChannels;
  midNumChannels = other.midNumChannels;
  regularNumChannels = other.regularNumChannels;
  gpoolNumChannels = other.gpoolNumChannels;
  initialConv = std::move(other.initialConv);
  initialMatMul = std::move(other.initialMatMul);
  blocks = std::move(other.blocks);
  trunkTipBN = std::move(other.trunkTipBN);
  trunkTipActivation = std::move(other.trunkTipActivation);
}

TrunkDesc& TrunkDesc::operator=(TrunkDesc&& other) {
  name = std::move(other.name);
  version = other.version;
  numBlocks = other.numBlocks;
  trunkNumChannels = other.trunkNumChannels;
  midNumChannels = other.midNumChannels;
  regularNumChannels = other.regularNumChannels;
  gpoolNumChannels = other.gpoolNumChannels;
  initialConv = std::move(other.initialConv);
  initialMatMul = std::move(other.initialMatMul);
  blocks = std::move(other.blocks);
  trunkTipBN = std::move(other.trunkTipBN);
  trunkTipActivation = std::move(other.trunkTipActivation);
  return *this;
}

void TrunkDesc::iterConvLayers(std::function<void(const ConvLayerDesc& desc)> f) const {
  f(initialConv);
  for(int i = 0; i < blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlockDesc* desc = (ResidualBlockDesc*)blocks[i].second.get();
      desc->iterConvLayers(f);
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlockDesc* desc = (GlobalPoolingResidualBlockDesc*)blocks[i].second.get();
      desc->iterConvLayers(f);
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlockDesc* desc = (NestedBottleneckResidualBlockDesc*)blocks[i].second.get();
      desc->iterConvLayers(f);
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND ||
            blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      // Transformer projections are matmuls, not convolutions.
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
}

static bool blocksContainTransformerRecursive(
  const vector<pair<int, unique_ptr_void>>& blocks
) {
  for(size_t i = 0; i < blocks.size(); i++) {
    if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND ||
       blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND)
      return true;
    if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      const NestedBottleneckResidualBlockDesc* desc =
        (const NestedBottleneckResidualBlockDesc*)blocks[i].second.get();
      if(blocksContainTransformerRecursive(desc->blocks))
        return true;
    }
  }
  return false;
}

bool TrunkDesc::hasAnyTransformerBlocks() const {
  return blocksContainTransformerRecursive(blocks);
}

//-----------------------------------------------------------------------------

PolicyHeadDesc::PolicyHeadDesc() : version(-1) {}

PolicyHeadDesc::PolicyHeadDesc(istream& in, int vrsn, bool binaryFloats) {
  in >> name;
  version = vrsn;

  if(in.fail())
    throw StringError(name + ": policy head failed to parse name");

  p1Conv = ConvLayerDesc(in,binaryFloats);
  g1Conv = ConvLayerDesc(in,binaryFloats);
  g1BN = BatchNormLayerDesc(in,binaryFloats);
  g1Activation = ActivationLayerDesc(in,version);
  gpoolToBiasMul = MatMulLayerDesc(in,binaryFloats);
  p1BN = BatchNormLayerDesc(in,binaryFloats);
  p1Activation = ActivationLayerDesc(in,version);
  p2Conv = ConvLayerDesc(in,binaryFloats);
  gpoolToPassMul = MatMulLayerDesc(in,binaryFloats);

  if(in.fail())
    throw StringError(name + ": policy head istream fail after parsing layers");

  if(p1Conv.outChannels != p1BN.numChannels)
    throw StringError(
      name +
      Global::strprintf(": p1Conv.outChannels (%d) != p1BN.numChannels (%d)", p1Conv.outChannels, p1BN.numChannels));
  if(g1Conv.outChannels != g1BN.numChannels)
    throw StringError(
      name +
      Global::strprintf(": g1Conv.outChannels (%d) != g1BN.numChannels (%d)", g1Conv.outChannels, g1BN.numChannels));
  if(gpoolToBiasMul.inChannels != g1BN.numChannels * 3)
    throw StringError(
      name + Global::strprintf(
               ": gpoolToBiasMul.inChannels (%d) != g1BN.numChannels*3 (%d)",
               gpoolToBiasMul.inChannels,
               g1BN.numChannels * 3));
  if(gpoolToBiasMul.outChannels != p1BN.numChannels)
    throw StringError(
      name +
      Global::strprintf(
        ": gpoolToBiasMul.outChannels (%d) != p1BN.numChannels (%d)", gpoolToBiasMul.outChannels, p1BN.numChannels));
  if(p2Conv.inChannels != p1BN.numChannels)
    throw StringError(
      name +
      Global::strprintf(": p2Conv.inChannels (%d) != p1BN.numChannels (%d)", p2Conv.inChannels, p1BN.numChannels));
  if(p2Conv.outChannels != 1)
    throw StringError(name + Global::strprintf(": p2Conv.outChannels (%d) != 1", p2Conv.outChannels));
  if(gpoolToPassMul.inChannels != g1BN.numChannels * 3)
    throw StringError(
      name + Global::strprintf(
               ": gpoolToPassMul.inChannels (%d) != g1BN.numChannels*3 (%d)",
               gpoolToPassMul.inChannels,
               g1BN.numChannels * 3));
  if(gpoolToPassMul.outChannels != 1)
    throw StringError(name + Global::strprintf(": gpoolToPassMul.outChannels (%d) != 1", gpoolToPassMul.outChannels));
}

PolicyHeadDesc::~PolicyHeadDesc() {}

PolicyHeadDesc::PolicyHeadDesc(PolicyHeadDesc&& other) {
  *this = std::move(other);
}

PolicyHeadDesc& PolicyHeadDesc::operator=(PolicyHeadDesc&& other) {
  name = std::move(other.name);
  version = other.version;
  p1Conv = std::move(other.p1Conv);
  g1Conv = std::move(other.g1Conv);
  g1BN = std::move(other.g1BN);
  g1Activation = std::move(other.g1Activation);
  gpoolToBiasMul = std::move(other.gpoolToBiasMul);
  p1BN = std::move(other.p1BN);
  p1Activation = std::move(other.p1Activation);
  p2Conv = std::move(other.p2Conv);
  gpoolToPassMul = std::move(other.gpoolToPassMul);
  return *this;
}

void PolicyHeadDesc::iterConvLayers(std::function<void(const ConvLayerDesc& desc)> f) const {
  f(p1Conv);
  f(g1Conv);
  f(p2Conv);
}

//-----------------------------------------------------------------------------

ValueHeadDesc::ValueHeadDesc() : version(-1) {}

ValueHeadDesc::ValueHeadDesc(istream& in, int vrsn, bool binaryFloats) {
  in >> name;
  version = vrsn;

  if(in.fail())
    throw StringError(name + ": value head failed to parse name");

  v1Conv = ConvLayerDesc(in,binaryFloats);
  v1BN = BatchNormLayerDesc(in,binaryFloats);
  v1Activation = ActivationLayerDesc(in,version);
  v2Mul = MatMulLayerDesc(in,binaryFloats);
  v2Bias = MatBiasLayerDesc(in,binaryFloats);
  v2Activation = ActivationLayerDesc(in,version);
  v3Mul = MatMulLayerDesc(in,binaryFloats);
  v3Bias = MatBiasLayerDesc(in,binaryFloats);

  sv3Mul = MatMulLayerDesc(in,binaryFloats);
  sv3Bias = MatBiasLayerDesc(in,binaryFloats);
  vOwnershipConv = ConvLayerDesc(in,binaryFloats);

  if(in.fail())
    throw StringError(name + ": value head istream fail after parsing layers");

  if(v1Conv.outChannels != v1BN.numChannels)
    throw StringError(
      name +
      Global::strprintf(": v1Conv.outChannels (%d) != v1BN.numChannels (%d)", v1Conv.outChannels, v1BN.numChannels));

  if(v2Mul.inChannels != v1BN.numChannels * 3)
    throw StringError(
      name + Global::strprintf(
               ": v2Mul.inChannels (%d) != v1BN.numChannels*3 (%d)", v2Mul.inChannels, v1BN.numChannels * 3));

  if(v2Mul.outChannels != v2Bias.numChannels)
    throw StringError(
      name +
      Global::strprintf(": v2Mul.outChannels (%d) != v2Bias.numChannels (%d)", v2Mul.outChannels, v2Bias.numChannels));
  if(v2Mul.outChannels != v3Mul.inChannels)
    throw StringError(
      name +
      Global::strprintf(": v2Mul.outChannels (%d) != v3Mul.inChannels (%d)", v2Mul.outChannels, v3Mul.inChannels));
  if(v3Mul.outChannels != 3)
    throw StringError(name + Global::strprintf(": v3Mul.outChannels (%d) != 3", v3Mul.outChannels));
  if(v3Bias.numChannels != 3)
    throw StringError(name + Global::strprintf(": v3Bias.numChannels (%d) != 3", v3Bias.numChannels));

  if(sv3Mul.inChannels != v2Mul.outChannels)
    throw StringError(
      name +
      Global::strprintf(": sv3Mul.inChannels (%d) != v2Mul.outChannels (%d)", sv3Mul.inChannels, v2Mul.outChannels));

  if(version >= 9) {
    if(sv3Mul.outChannels != 6)
      throw StringError(name + Global::strprintf(": sv3Mul.outChannels (%d) != 6", sv3Mul.outChannels));
    if(sv3Bias.numChannels != 6)
      throw StringError(name + Global::strprintf(": sv3Bias.numChannels (%d) != 6", sv3Bias.numChannels));
  }
  else if(version >= 8) {
    if(sv3Mul.outChannels != 4)
      throw StringError(name + Global::strprintf(": sv3Mul.outChannels (%d) != 4", sv3Mul.outChannels));
    if(sv3Bias.numChannels != 4)
      throw StringError(name + Global::strprintf(": sv3Bias.numChannels (%d) != 4", sv3Bias.numChannels));
  }
  else if(version >= 4) {
    if(sv3Mul.outChannels != 2)
      throw StringError(name + Global::strprintf(": sv3Mul.outChannels (%d) != 2", sv3Mul.outChannels));
    if(sv3Bias.numChannels != 2)
      throw StringError(name + Global::strprintf(": sv3Bias.numChannels (%d) != 2", sv3Bias.numChannels));
  }
  else {
    if(sv3Mul.outChannels != 1)
      throw StringError(name + Global::strprintf(": sv3Mul.outChannels (%d) != 1", sv3Mul.outChannels));
    if(sv3Bias.numChannels != 1)
      throw StringError(name + Global::strprintf(": sv3Bias.numChannels (%d) != 1", sv3Bias.numChannels));
  }

  if(vOwnershipConv.inChannels != v1Conv.outChannels)
    throw StringError(
      name + Global::strprintf(
               ": vOwnershipConv.outChannels (%d) != v1Conv.outChannels (%d)",
               vOwnershipConv.inChannels,
               v1Conv.outChannels));
  if(vOwnershipConv.outChannels != 1)
    throw StringError(name + Global::strprintf(": vOwnershipConv.outChannels (%d) != 1", vOwnershipConv.outChannels));
}

ValueHeadDesc::~ValueHeadDesc() {}

ValueHeadDesc::ValueHeadDesc(ValueHeadDesc&& other) {
  *this = std::move(other);
}

ValueHeadDesc& ValueHeadDesc::operator=(ValueHeadDesc&& other) {
  name = std::move(other.name);
  version = other.version;
  v1Conv = std::move(other.v1Conv);
  v1BN = std::move(other.v1BN);
  v1Activation = std::move(other.v1Activation);
  v2Mul = std::move(other.v2Mul);
  v2Bias = std::move(other.v2Bias);
  v2Activation = std::move(other.v2Activation);
  v3Mul = std::move(other.v3Mul);
  v3Bias = std::move(other.v3Bias);
  sv3Mul = std::move(other.sv3Mul);
  sv3Bias = std::move(other.sv3Bias);
  vOwnershipConv = std::move(other.vOwnershipConv);
  return *this;
}

void ValueHeadDesc::iterConvLayers(std::function<void(const ConvLayerDesc& desc)> f) const {
  f(v1Conv);
  f(vOwnershipConv);
}

//-----------------------------------------------------------------------------

ModelDesc::ModelDesc()
  : version(-1),
    numInputChannels(0),
    numInputGlobalChannels(0),
    numValueChannels(0),
    numScoreValueChannels(0),
    numOwnershipChannels(0) {}

ModelDesc::ModelDesc(istream& in, const string& sha256_, bool binaryFloats) {
  in >> name;
  sha256 = sha256_;
  in >> version;
  if(in.fail())
    throw StringError("Model failed to parse name or version. Is this a valid model file? You probably specified the wrong file.");

  if(version < 0)
    throw StringError("This neural net has an invalid version, you probably specified the wrong file. Supposed model version: " + Global::intToString(version));
  if(version < 3)
    throw StringError("This neural net is from an extremely old version of KataGo and is no longer supported by the engine. Model version: " + Global::intToString(version));
  if(version > NNModelVersion::latestModelVersionImplemented)
    throw StringError("This neural net requires a newer KataGo version. Obtain a newer KataGo at https://github.com/lightvector/KataGo. Model version: " + Global::intToString(version));

  in >> numInputChannels;
  if(in.fail())
    throw StringError(name + ": model failed to parse numInputChannels");
  if(numInputChannels <= 0)
    throw StringError(name + ": model numInputChannels must be positive");

  in >> numInputGlobalChannels;
  if(in.fail())
    throw StringError(name + ": model failed to parse numInputGlobalChannels");
  if(numInputGlobalChannels <= 0)
    throw StringError(name + ": model numInputGlobalChannels must be positive");

  trunk = TrunkDesc(in, version, binaryFloats);
  policyHead = PolicyHeadDesc(in, version, binaryFloats);
  valueHead = ValueHeadDesc(in, version, binaryFloats);

  numValueChannels = valueHead.v3Mul.outChannels;
  numScoreValueChannels = valueHead.sv3Mul.outChannels;
  numOwnershipChannels = valueHead.vOwnershipConv.outChannels;

  if(in.fail())
    throw StringError(name + ": model desc istream fail after parsing model");

  if(numInputChannels != trunk.initialConv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": numInputChannels (%d) != trunk.initialConv.inChannels (%d)",
               numInputChannels,
               trunk.initialConv.inChannels));
  if(numInputGlobalChannels != trunk.initialMatMul.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": numInputChannels (%d) != trunk.initialMatMul.inChannels (%d)",
               numInputGlobalChannels,
               trunk.initialMatMul.inChannels));

  if(trunk.trunkNumChannels != policyHead.p1Conv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": trunk.trunkNumChannels (%d) != policyHead.p1Conv.inChannels (%d)",
               trunk.trunkNumChannels,
               policyHead.p1Conv.inChannels));
  if(trunk.trunkNumChannels != policyHead.g1Conv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": trunk.trunkNumChannels (%d) != policyHead.g1Conv.inChannels (%d)",
               trunk.trunkNumChannels,
               policyHead.g1Conv.inChannels));
  if(trunk.trunkNumChannels != valueHead.v1Conv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": trunk.trunkNumChannels (%d) != valueHead.v1Conv.inChannels (%d)",
               trunk.trunkNumChannels,
               valueHead.v1Conv.inChannels));
}

ModelDesc::~ModelDesc() {}

ModelDesc::ModelDesc(ModelDesc&& other) {
  *this = std::move(other);
}

ModelDesc& ModelDesc::operator=(ModelDesc&& other) {
  name = std::move(other.name);
  sha256 = std::move(other.sha256);
  version = other.version;
  numInputChannels = other.numInputChannels;
  numInputGlobalChannels = other.numInputGlobalChannels;
  numValueChannels = other.numValueChannels;
  numScoreValueChannels = other.numScoreValueChannels;
  numOwnershipChannels = other.numOwnershipChannels;
  trunk = std::move(other.trunk);
  policyHead = std::move(other.policyHead);
  valueHead = std::move(other.valueHead);
  return *this;
}

void ModelDesc::iterConvLayers(std::function<void(const ConvLayerDesc& desc)> f) const {
  trunk.iterConvLayers(f);
  policyHead.iterConvLayers(f);
  valueHead.iterConvLayers(f);
}

int ModelDesc::maxConvChannels(int convXSize, int convYSize) const {
  int c = 0;
  auto f = [&c,convXSize,convYSize](const ConvLayerDesc& desc) {
    if(desc.convXSize == convXSize && desc.convYSize == convYSize) {
      if(desc.inChannels > c)
        c = desc.inChannels;
      if(desc.outChannels > c)
        c = desc.outChannels;
    }
  };
  iterConvLayers(f);
  return c;
}

struct NonCopyingStreamBuf : public std::streambuf
{
  NonCopyingStreamBuf(string& str) {
    char* s = &str[0];
    size_t n = str.size();
    setg(s, s, s + n);
  }
};

void ModelDesc::loadFromFileMaybeGZipped(const string& fileName, ModelDesc& descBuf, const string& expectedSha256) {
  try {
    string lower = Global::toLower(fileName);
    //Read model file with no compression if it's directly named .txt or .bin
    if(Global::isSuffix(lower,".txt")) {
      bool binaryFloats = false;
      string uncompressed;
      string sha256Buf;
      FileUtils::loadFileIntoString(fileName, expectedSha256, uncompressed, &sha256Buf);
      NonCopyingStreamBuf uncompressedStreamBuf(uncompressed);
      std::istream uncompressedIn(&uncompressedStreamBuf);
      descBuf = ModelDesc(uncompressedIn, sha256Buf, binaryFloats);
    }
    else if(Global::isSuffix(lower,".bin")) {
      bool binaryFloats = true;
      string uncompressed;
      string sha256Buf;
      FileUtils::loadFileIntoString(fileName, expectedSha256, uncompressed, &sha256Buf);
      NonCopyingStreamBuf uncompressedStreamBuf(uncompressed);
      std::istream uncompressedIn(&uncompressedStreamBuf);
      descBuf = ModelDesc(uncompressedIn, sha256Buf, binaryFloats);
    }
    else if(Global::isSuffix(lower,".txt.gz") || Global::isSuffix(lower,".bin.gz") || Global::isSuffix(lower,".gz")) {
      string uncompressed;
      string sha256Buf;
      FileUtils::uncompressAndLoadFileIntoString(fileName, expectedSha256, uncompressed, &sha256Buf);

      bool binaryFloats = !Global::isSuffix(lower,".txt.gz");
      try {
        //Now, initialize an istream to read from the string
        NonCopyingStreamBuf uncompressedStreamBuf(uncompressed);
        std::istream uncompressedIn(&uncompressedStreamBuf);
        //And read in the model desc
        descBuf = ModelDesc(uncompressedIn, sha256Buf, binaryFloats);
      }
      catch(const StringError& e) {
        //On failure, try again to read as a .txt.gz file if the extension was ambiguous
        bool tryAgain = binaryFloats && !Global::isSuffix(lower,".bin.gz");
        if(!tryAgain)
          throw;
        else {
          binaryFloats = false;
          try {
            NonCopyingStreamBuf uncompressedStreamBuf(uncompressed);
            std::istream uncompressedIn(&uncompressedStreamBuf);
            descBuf = ModelDesc(uncompressedIn, sha256Buf, binaryFloats);
          }
          catch(const StringError& e2) {
            throw StringError(string("Could neither parse .gz model as .txt.gz model nor as .bin.gz model, errors were:\n") + e2.what() + "\n" + e.what());
          }
        }
      }
    }
    else {
      throw StringError("Model file should end with .txt, .bin, .txt.gz, .bin.gz, or possibly just .gz. (If it doesn't have one of these extensions already, it's probably the wrong file, renaming will probably NOT help).");
    }
  }
  catch(const StringError& e) {
    throw StringError("Error loading or parsing model file " + fileName + ": " + e.what());
  }
}
void ModelDesc::loadFromONNX(const string& onnxFile, ModelDesc& descBuf) {
  descBuf.onnxHeader.load(onnxFile);

  descBuf.version = descBuf.onnxHeader.modelVersion;
  descBuf.name = descBuf.onnxHeader.modelName;
  descBuf.numInputChannels = descBuf.onnxHeader.num_spatial_inputs;
  descBuf.numInputGlobalChannels = descBuf.onnxHeader.num_global_inputs;
  if(descBuf.numInputChannels != NNModelVersion::getNumSpatialFeatures(descBuf.version))
    throw StringError("ONNX model requires num_spatial_inputs metadata field to match modelVersion");
  if(descBuf.numInputGlobalChannels != NNModelVersion::getNumGlobalFeatures(descBuf.version))
    throw StringError("ONNX model requires num_global_inputs metadata field to match modelVersion");

  //descBuf.numPolicyChannels = 0;  // will not be used
  descBuf.numValueChannels = 0;   // will not be used
  descBuf.numScoreValueChannels = 0;
  descBuf.numOwnershipChannels = 0;
}

Rules ModelDesc::getSupportedRules(const Rules& desiredRules, bool& supported) const {
  static_assert(NNModelVersion::latestModelVersionImplemented == 103, "");
  Rules rules = desiredRules;
  supported = true;
  if(version <= 103) {
  }
  else {
    ASSERT_UNREACHABLE;
  }

  return rules;
}
