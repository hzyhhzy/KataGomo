#include "fa4.h"

#include <cmath>
#include <cstring>

namespace C384H12Fa4Sm120 {

namespace {

bool isQualifiedBatch(int batch) {
  return batch == 24 || batch == 28;
}

bool accessorsMatch(const Candidate& candidate) {
  return candidate.compiledBatch != nullptr &&
    candidate.compiledSequence != nullptr &&
    candidate.compiledHeads != nullptr &&
    candidate.compiledHeadDim != nullptr &&
    candidate.compiledTileM != nullptr &&
    candidate.compiledTileN != nullptr &&
    candidate.compiledNumStages != nullptr &&
    candidate.compiledNumWarps != nullptr &&
    candidate.compiledInputLayout != nullptr &&
    candidate.compiledAccumulation != nullptr &&
    candidate.compiledId != nullptr &&
    candidate.compiledBatch() == candidate.batch &&
    candidate.compiledSequence() == candidate.sequence &&
    candidate.compiledHeads() == candidate.heads &&
    candidate.compiledHeadDim() == candidate.headDim &&
    candidate.compiledTileM() == candidate.tileM &&
    candidate.compiledTileN() == candidate.tileN &&
    candidate.compiledNumStages() == candidate.numStages &&
    candidate.compiledNumWarps() == candidate.numWarps &&
    candidate.compiledInputLayout() == static_cast<int>(candidate.inputLayout) &&
    candidate.compiledAccumulation() == static_cast<int>(candidate.accumulation) &&
    candidate.compiledId() != nullptr &&
    std::strcmp(candidate.compiledId(),candidate.id) == 0;
}

bool validLayout(InputLayout layout) {
  return layout == InputLayout::PlanarQkv ||
    layout == InputLayout::PackedTokenQkv;
}

bool validAccumulation(Accumulation accumulation) {
  return accumulation == Accumulation::Fp32 ||
    accumulation == Accumulation::QkFp16 ||
    accumulation == Accumulation::PvFp16 ||
    accumulation == Accumulation::BothFp16;
}

}  // namespace

bool candidateWellFormed(const Candidate& candidate) {
  return candidate.abiVersion == kRegistryAbiVersion &&
    isQualifiedBatch(candidate.batch) &&
    candidate.sequence == kSequenceLength &&
    candidate.heads == kHeads && candidate.headDim == kHeadDim &&
    (candidate.tileM == 64 || candidate.tileM == 128) &&
    (candidate.tileN == 64 || candidate.tileN == 96 || candidate.tileN == 128) &&
    (candidate.numStages == 1 || candidate.numStages == 2) &&
    (candidate.numWarps == 4 || candidate.numWarps == 8) &&
    validLayout(candidate.inputLayout) &&
    validAccumulation(candidate.accumulation) &&
    candidate.id != nullptr && candidate.id[0] != '\0' &&
    candidate.prepare != nullptr && candidate.launch != nullptr;
}

bool registryWellFormed(const RegistryView& registry) {
  if(registry.mode == ArtifactMode::Disabled)
    return registry.candidates == nullptr && registry.count == 0;
  if(registry.candidates == nullptr)
    return false;
  if(registry.mode == ArtifactMode::BatchSearch) {
    if(registry.count != 2)
      return false;
  }
  else if(registry.mode == ArtifactMode::Production) {
    if(registry.count != 1)
      return false;
  }
  else
    return false;

  bool saw24 = false;
  bool saw28 = false;
  int searchTileM = 0;
  int searchTileN = 0;
  int searchStages = 0;
  int searchWarps = 0;
  InputLayout searchLayout = InputLayout::PlanarQkv;
  Accumulation searchAccumulation = Accumulation::Fp32;
  for(std::size_t i = 0; i < registry.count; i++) {
    const Candidate& candidate = registry.candidates[i];
    if(!candidateWellFormed(candidate))
      return false;
    for(std::size_t j = 0; j < i; j++) {
      if(registry.candidates[j].batch == candidate.batch ||
         std::strcmp(registry.candidates[j].id,candidate.id) == 0)
        return false;
    }
    saw24 = saw24 || candidate.batch == 24;
    saw28 = saw28 || candidate.batch == 28;
    if(i == 0) {
      searchTileM = candidate.tileM;
      searchTileN = candidate.tileN;
      searchStages = candidate.numStages;
      searchWarps = candidate.numWarps;
      searchLayout = candidate.inputLayout;
      searchAccumulation = candidate.accumulation;
    }
    else if(registry.mode == ArtifactMode::BatchSearch &&
            (candidate.tileM != searchTileM || candidate.tileN != searchTileN ||
             candidate.numStages != searchStages ||
             candidate.numWarps != searchWarps ||
             candidate.inputLayout != searchLayout ||
             candidate.accumulation != searchAccumulation))
      return false;
  }
  return registry.mode != ArtifactMode::BatchSearch || (saw24 && saw28);
}

const Candidate* findExactCandidate(const RegistryView& registry, int batch) {
  if(!registryWellFormed(registry))
    return nullptr;
  for(std::size_t i = 0; i < registry.count; i++) {
    if(registry.candidates[i].batch == batch)
      return &registry.candidates[i];
  }
  return nullptr;
}

bool supportsExactBatch(int batch) {
  return findExactCandidate(generatedRegistry(),batch) != nullptr;
}

bool proofCompatible(
  const PreparedProof& proof,
  int batch,
  int deviceOrdinal,
  InputLayout requiredLayout
) {
  return proof.abiVersion == kPreparedProofAbiVersion &&
    proof.batch == batch && proof.sequence == kSequenceLength &&
    proof.heads == kHeads && proof.headDim == kHeadDim &&
    proof.deviceOrdinal == deviceOrdinal &&
    proof.inputLayout == requiredLayout &&
    proof.id != nullptr && proof.id[0] != '\0' &&
    proof.launch != nullptr && proof.implementationCookie ==
      reinterpret_cast<uintptr_t>(proof.launch);
}

cudaError_t prepareProofForExactBatch(
  int batch,
  int deviceOrdinal,
  InputLayout requiredLayout,
  PreparedProof& proof
) {
  proof = PreparedProof{};
  if(deviceOrdinal < 0 || !validLayout(requiredLayout))
    return cudaErrorInvalidValue;
  const RegistryView registry = generatedRegistry();
  if(!registryWellFormed(registry))
    return cudaErrorInvalidValue;
  const Candidate* candidate = findExactCandidate(registry,batch);
  if(candidate == nullptr || candidate->inputLayout != requiredLayout)
    return cudaErrorNotSupported;
  if(!accessorsMatch(*candidate))
    return cudaErrorInvalidValue;
  const cudaError_t status = candidate->prepare(deviceOrdinal);
  if(status != cudaSuccess)
    return status;
  proof.abiVersion = kPreparedProofAbiVersion;
  proof.batch = candidate->batch;
  proof.sequence = candidate->sequence;
  proof.heads = candidate->heads;
  proof.headDim = candidate->headDim;
  proof.deviceOrdinal = deviceOrdinal;
  proof.inputLayout = candidate->inputLayout;
  proof.id = candidate->id;
  proof.launch = candidate->launch;
  proof.implementationCookie = reinterpret_cast<uintptr_t>(candidate->launch);
  return cudaSuccess;
}

LaunchResult launch(
  half* q,
  half* k,
  half* v,
  half* output,
  int batch,
  int sequence,
  int heads,
  int kvHeads,
  int qHeadDim,
  int vHeadDim,
  bool usingFp16,
  bool usingNhwc,
  InputLayout inputLayout,
  const void* mask,
  bool recipeEligibleC384,
  const PreparedProof* preparedProof,
  int deviceOrdinal,
  int computeMajor,
  int computeMinor,
  cudaStream_t stream
) {
  if(!recipeEligibleC384 || computeMajor != 12 || computeMinor != 0 ||
     deviceOrdinal < 0 ||
     !usingFp16 || !usingNhwc || !validLayout(inputLayout) || mask != nullptr ||
     sequence != kSequenceLength || heads != kHeads || kvHeads != kHeads ||
     qHeadDim != kHeadDim || vHeadDim != kHeadDim)
    return {false,cudaSuccess,nullptr};

  // A different actual batch/layout is a pre-enqueue capability miss. This is
  // essential for a B28 handle evaluating an actual B24 tail: it must use the
  // planar fallback rather than looking up B24 and then hard-failing a B28
  // proof. Exact packed QKV callers gate this before producing packed bytes.
  if(preparedProof == nullptr || preparedProof->batch != batch ||
     preparedProof->inputLayout != inputLayout)
    return {false,cudaSuccess,nullptr};
  if(!proofCompatible(*preparedProof,batch,deviceOrdinal,inputLayout) ||
     q == nullptr || k == nullptr || v == nullptr || output == nullptr)
    return {true,cudaErrorInvalidValue,preparedProof->id};

  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  return {
    true,
    preparedProof->launch(
      q,k,v,output,batch,sequence,heads,qHeadDim,scale,
      static_cast<uint32_t>(inputLayout),deviceOrdinal,stream),
    preparedProof->id,
  };
}

}  // namespace C384H12Fa4Sm120
