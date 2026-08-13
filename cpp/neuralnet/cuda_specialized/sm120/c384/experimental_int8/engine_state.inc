  void configureC384Int8Experiment(
    bool int8RuntimeEnabled,
    const ModelDesc& modelDesc
  ) {
    c384Int8Mode = C384Int8Experiment::parseEngineMode(
      std::getenv("KATAGO_C384_INT8_MODE"));
    c384Int8CandidateEligible = false;
    c384Int8TransactionEnabled = false;
    expectedC384Int8Attention = 0;
    expectedC384Int8Ffn = 0;
    expectedC384Int8Down = 0;
    if(c384Int8Mode == C384Int8Experiment::EngineMode::Invalid) {
      if(logger != NULL)
        logger->write(
          "KATAGO_C384_SM120_INT8_UNAVAILABLE reason=invalid-runtime-mode "
          "fallback=prepared-fp16 expected=off|conservative|aggressive");
      return;
    }
    C384Int8Experiment::EngineEligibility shape;
    shape.modelVersion = modelDesc.version;
    shape.batchSize = c384ExactBatchSize;
    shape.boardX = C384ExactFixedAot::kBoardX;
    shape.boardY = C384ExactFixedAot::kBoardY;
    shape.modelDepth = c384ExactModelDepth;
    shape.attentionBlocks = c384ExactAttentionBlockCount;
    shape.ffnBlocks = c384ExactFfnBlockCount;
    shape.channels = C384ExactFixedAot::kChannels;
    shape.heads = C384ExactFixedAot::kNumHeads;
    shape.headDim = C384ExactFixedAot::kHeadDim;
    shape.ffnChannels = C384ExactFixedAot::kFfnChannels;
    shape.computeCapability =
      (uint32_t)(majorComputeCapability * 10 + minorComputeCapability);
    shape.alternatingAttentionFfn = true;
    shape.usingFp16 = true;
    shape.usingNhwc = true;
    shape.exactNoMask = true;
    shape.learnedRope = true;
    shape.qkNorm = c384ExactQkNormModel;
    shape.swigluClipBits = c384ExactSwigluClipBits;
    const bool requested =
      c384Int8Mode == C384Int8Experiment::EngineMode::Conservative ||
      c384Int8Mode == C384Int8Experiment::EngineMode::Aggressive;
    c384Int8CandidateEligible = requested && int8RuntimeEnabled &&
      c384ExactModelEligible && C384Int8Experiment::engineShapeEligible(shape);
    if(c384Int8CandidateEligible) {
      expectedC384Int8Attention = c384ExactAttentionBlockCount;
      expectedC384Int8Ffn = c384ExactFfnBlockCount;
      expectedC384Int8Down =
        c384Int8Mode == C384Int8Experiment::EngineMode::Aggressive ?
        c384ExactFfnBlockCount : 0;
    }
    if(logger != NULL) {
      logger->write(
        string("KATAGO_C384_SM120_INT8_RUNTIME_GATE requested=") +
        (requested ? "1" : "0") + " runtime_int8=" +
        (int8RuntimeEnabled ? "1" : "0") + " mode=" +
        C384Int8Experiment::engineModeName(c384Int8Mode) + " eligible=" +
        (c384Int8CandidateEligible ? "1" : "0") +
        " weights=v105-load-time-quant default=off");
      if(requested && !c384Int8CandidateEligible)
        logger->write(
          "KATAGO_C384_SM120_INT8_UNAVAILABLE reason=eligibility-mismatch "
          "fallback=prepared-fp16");
    }
  }

  C384Int8Experiment::ProjectionMode c384Int8ProjectionMode() const noexcept {
    return c384Int8Mode == C384Int8Experiment::EngineMode::Conservative ?
      C384Int8Experiment::ProjectionMode::ConservativeQk :
      C384Int8Experiment::ProjectionMode::AggressiveQkv;
  }

  void registerC384Int8PreparedCleanup(std::function<void()> cleanup) {
    c384Int8PreparedCleanupRegistry.push_back(std::move(cleanup));
  }

  void disableC384Int8Experiment(const string& reason) {
    if(!c384Int8CandidateEligible && !c384Int8TransactionEnabled &&
       c384Int8PreparedCleanupRegistry.empty())
      return;
    const int beforeAttention = preparedC384Int8Attention;
    const int beforeFfn = preparedC384Int8Ffn;
    const int beforeDown = preparedC384Int8Down;
    c384Int8CandidateEligible = false;
    c384Int8TransactionEnabled = false;
    for(auto cleanup = c384Int8PreparedCleanupRegistry.rbegin();
        cleanup != c384Int8PreparedCleanupRegistry.rend(); ++cleanup)
      (*cleanup)();
    c384Int8PreparedCleanupRegistry.clear();
    preparedC384Int8Attention = 0;
    preparedC384Int8Ffn = 0;
    preparedC384Int8Down = 0;
    activeC384Int8Attention = 0;
    activeC384Int8Ffn = 0;
    activeC384Int8Down = 0;
    if(logger != NULL)
      logger->write(
        string("KATAGO_C384_SM120_INT8_UNAVAILABLE reason=") + reason +
        " prepared_attention=" + Global::intToString(beforeAttention) +
        "/" + Global::intToString(expectedC384Int8Attention) +
        " prepared_ffn=" + Global::intToString(beforeFfn) + "/" +
        Global::intToString(expectedC384Int8Ffn) + " prepared_down=" +
        Global::intToString(beforeDown) + "/" +
        Global::intToString(expectedC384Int8Down) +
        " fallback=prepared-fp16");
  }

  bool c384Int8PreparedComplete() const noexcept {
    return c384Int8CandidateEligible &&
      preparedC384Int8Attention == expectedC384Int8Attention &&
      preparedC384Int8Ffn == expectedC384Int8Ffn &&
      preparedC384Int8Down == expectedC384Int8Down;
  }

  void commitC384Int8Experiment(bool exactReady, bool scratchReady) {
    if(!c384Int8CandidateEligible)
      return;
    const bool ready =
      c384Int8PreparedComplete() && exactReady && scratchReady;
    if(logger != NULL)
      logger->write(
        string("KATAGO_C384_SM120_INT8_PREPARED mode=") +
        C384Int8Experiment::engineModeName(c384Int8Mode) + " attention=" +
        Global::intToString(preparedC384Int8Attention) + "/" +
        Global::intToString(expectedC384Int8Attention) + " ffn=" +
        Global::intToString(preparedC384Int8Ffn) + "/" +
        Global::intToString(expectedC384Int8Ffn) + " down=" +
        (expectedC384Int8Down == 0 ? string("fp16") :
          Global::intToString(preparedC384Int8Down) + "/" +
          Global::intToString(expectedC384Int8Down)) +
        " exact_dependencies=" + (exactReady ? "1" : "0") +
        " scratch=" + (scratchReady ? "1" : "0") +
        " transaction=" + (ready ? "active" : "fallback"));
    if(!ready) {
      disableC384Int8Experiment("transaction-incomplete");
      return;
    }
    c384Int8TransactionEnabled = true;
    c384Int8PreparedCleanupRegistry.clear();
  }

  void noteC384Int8Launch(bool& blockCounted, int& activeCount) {
    if(!c384Int8TransactionEnabled || blockCounted)
      return;
    blockCounted = true;
    activeCount++;
    const bool complete =
      activeC384Int8Attention == expectedC384Int8Attention &&
      activeC384Int8Ffn == expectedC384Int8Ffn &&
      activeC384Int8Down == expectedC384Int8Down;
    if(complete && !loggedC384Int8TransactionActive && logger != NULL) {
      const auto p = static_cast<C384Int8Experiment::ProjectionTactic>(
        KATAGO_C384_INT8_PROJECTION_TACTIC);
      const auto d = static_cast<C384Int8Experiment::DualFfnTactic>(
        KATAGO_C384_INT8_DUAL_TACTIC);
      const auto w = static_cast<C384Int8Experiment::DownTactic>(
        KATAGO_C384_INT8_DOWN_TACTIC);
      logger->write(
        string("KATAGO_C384_SM120_INT8_ACTIVE model=v105 mode=") +
        C384Int8Experiment::engineModeName(c384Int8Mode) +
        " depth=36 attention=" + Global::intToString(activeC384Int8Attention) +
        "/36 dual=" + Global::intToString(activeC384Int8Ffn) +
        "/36 down=" + (expectedC384Int8Down == 0 ? string("fp16") :
          Global::intToString(activeC384Int8Down) + "/36") +
        " qknorm=1 clip=7 weights=v105-load-time-quant projection=" +
        C384Int8Experiment::projectionTacticName(p) + " dual_tactic=" +
        C384Int8Experiment::dualFfnTacticName(d) + " down_tactic=" +
        (expectedC384Int8Down == 0 ? string("fp16") :
          C384Int8Experiment::downTacticName(w)) +
        " product_quant=" + (expectedC384Int8Down == 0 ?
          string("none") : string("separate-v1")) +
        " scratch=persistent-per-handle transaction=active");
      loggedC384Int8TransactionActive = true;
    }
  }
