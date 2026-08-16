#ifndef FORBIDDEN_POINT_FINDER_BULK_TEST_H_
#define FORBIDDEN_POINT_FINDER_BULK_TEST_H_

#include <cstdint>

class CForbiddenPointFinder;

namespace ForbiddenPointFinderBulkTest {

// Exposes the production early-rejection bitmap for differential tests only.
void fillCandidateMap(const CForbiddenPointFinder& finder, uint8_t* candidateMap);

}

#endif
