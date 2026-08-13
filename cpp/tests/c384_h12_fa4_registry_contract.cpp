#include "../neuralnet/c384_h12_fa4_sm120.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace C384H12Fa4Sm120;

namespace {

int preparedDevice = -1;
int launchCount = 0;

cudaError_t prepare(int device) {
  preparedDevice = device;
  return device >= 0 ? cudaSuccess : cudaErrorInvalidDevice;
}

cudaError_t launch(
  void*, void*, void*, void*, int, int, int, int, float,
  uint32_t, cudaStream_t
) {
  launchCount++;
  return cudaSuccess;
}

#define DEFINE_ACCESSORS(NAME,BATCH,ID) \
  int NAME##_batch() { return BATCH; } \
  int NAME##_sequence() { return 225; } \
  int NAME##_heads() { return 12; } \
  int NAME##_dim() { return 32; } \
  const char* NAME##_id() { return ID; }

DEFINE_ACCESSORS(b24,24,"tm128-tn64-s1-w4-fp32-planar-b24-s225-h12-d32")
DEFINE_ACCESSORS(b28,28,"tm128-tn64-s1-w4-fp32-planar-b28-s225-h12-d32")
DEFINE_ACCESSORS(p28,28,"tm128-tn64-s1-w4-fp32-packed-token-b28-s225-h12-d32")

Candidate candidates[] = {
  {kRegistryAbiVersion,24,225,12,32,128,64,1,4,
   InputLayout::PlanarQkv,Accumulation::Fp32,b24_id(),
   b24_batch,b24_sequence,b24_heads,b24_dim,b24_id,prepare,launch},
  {kRegistryAbiVersion,28,225,12,32,128,64,1,4,
   InputLayout::PlanarQkv,Accumulation::Fp32,b28_id(),
   b28_batch,b28_sequence,b28_heads,b28_dim,b28_id,prepare,launch},
};

RegistryView view = {ArtifactMode::BatchSearch,candidates,2};

void require(bool condition, const char* message) {
  if(!condition) {
    std::cerr << "C384_H12_FA4_CONTRACT_FAILED: " << message << std::endl;
    std::exit(1);
  }
}

}  // namespace

namespace C384H12Fa4Sm120 {
RegistryView generatedRegistry() { return view; }
}  // namespace C384H12Fa4Sm120

int main() {
  require(registryWellFormed(view),"valid B24/B28 search registry rejected");
  require(findExactCandidate(view,24) == &candidates[0],"B24 dispatch miss");
  require(findExactCandidate(view,28) == &candidates[1],"B28 dispatch miss");
  require(findExactCandidate(view,40) == nullptr,"B40 evidence entered final search");

  RegistryView production = {ArtifactMode::Production,&candidates[1],1};
  require(registryWellFormed(production),"one-candidate production rejected");
  Candidate mismatchedTile = candidates[1];
  mismatchedTile.tileN = 96;
  Candidate mixed[] = {candidates[0],mismatchedTile};
  require(!registryWellFormed({ArtifactMode::BatchSearch,mixed,2}),
          "mixed tile batch comparison accepted");
  Candidate packed = candidates[1];
  packed.inputLayout = InputLayout::PackedTokenQkv;
  packed.id = p28_id();
  packed.compiledId = p28_id;
  Candidate mixedLayout[] = {candidates[0],packed};
  require(!registryWellFormed({ArtifactMode::BatchSearch,mixedLayout,2}),
          "mixed layout batch comparison accepted");

  PreparedProof proof;
  require(prepareProofForExactBatch(
            28,0,InputLayout::PlanarQkv,proof) == cudaSuccess,
          "B28 proof preparation failed");
  require(preparedDevice == 0,"proof did not bind device");
  require(proofCompatible(proof,28,0,InputLayout::PlanarQkv),
          "prepared proof rejected");
  require(!proofCompatible(proof,24,0,InputLayout::PlanarQkv),
          "proof crossed batch");
  require(!proofCompatible(proof,28,0,InputLayout::PackedTokenQkv),
          "proof crossed layout");

  // A packed-token exact QKV producer gets a separately prepared immutable
  // proof. Planar and packed layouts are never inferred from pointer values.
  view = {ArtifactMode::Production,&packed,1};
  PreparedProof packedProof;
  require(prepareProofForExactBatch(
            28,0,InputLayout::PackedTokenQkv,packedProof) == cudaSuccess,
          "packed-token proof preparation failed");
  require(proofCompatible(packedProof,28,0,InputLayout::PackedTokenQkv),
          "packed-token proof rejected");
  require(!proofCompatible(packedProof,28,0,InputLayout::PlanarQkv),
          "packed-token proof crossed into planar layout");
  view = {ArtifactMode::BatchSearch,candidates,2};

  half* q = reinterpret_cast<half*>(0x1000);
  half* k = reinterpret_cast<half*>(0x2000);
  half* v = reinterpret_cast<half*>(0x3000);
  half* o = reinterpret_cast<half*>(0x4000);
  const LaunchResult ok = C384H12Fa4Sm120::launch(
    q,k,v,o,28,225,12,12,32,32,true,true,InputLayout::PlanarQkv,
    nullptr,true,&proof,12,0,nullptr);
  require(ok.launched() && launchCount == 1,"default stream launch rejected");
  const LaunchResult neighbor = C384H12Fa4Sm120::launch(
    q,k,v,o,27,225,12,12,32,32,true,true,InputLayout::PlanarQkv,
    nullptr,true,&proof,12,0,nullptr);
  require(!neighbor.attempted,"neighbor batch did not fall back before enqueue");
  const LaunchResult wrongLayout = C384H12Fa4Sm120::launch(
    q,k,v,o,28,225,12,12,32,32,true,true,InputLayout::PackedTokenQkv,
    nullptr,true,&proof,12,0,nullptr);
  require(!wrongLayout.attempted,"layout mismatch did not fall back before enqueue");
  PreparedProof stale = proof;
  stale.implementationCookie++;
  const LaunchResult staleResult = C384H12Fa4Sm120::launch(
    q,k,v,o,28,225,12,12,32,32,true,true,InputLayout::PlanarQkv,
    nullptr,true,&stale,12,0,nullptr);
  require(staleResult.attempted && staleResult.status == cudaErrorInvalidValue,
          "owned stale proof did not fail hard");

  std::cout << "C384 H12 FA4 registry contract PASS" << std::endl;
  return 0;
}
