# MCTS New Implementation

这是一个基于 `mcts_old` 算法的新MCTS实现，增加了必胜必败判断和缓存优化功能。

## 主要特性

### 1. 必胜必败判断
- 每个节点记录是否已确定必胜或必败状态
- 记录到达胜负的步数
- 避免对结论已确定的节点继续搜索
- 优先选择必胜的走法

### 2. 缓存哈希表
缓存以下计算结果以提高性能：
- `getAllVCFAttackOrDefenseLocs` 的 `maybeWinner` 和 `gameEndMovenum`
- NNUE 输出的 `value` 结果
- `getLegalMovesWithPolicy` 的结果

### 3. 智能搜索策略
- 自动检测游戏结束状态
- 基于VCF分析进行必胜必败判断
- 结合NNUE评估和传统MCTS选择

## 文件结构

```
mcts/
├── MCTSsearch.h      # 头文件定义
├── MCTSsearch.cpp    # 主要实现
├── Makefile              # 编译配置
└── README.md             # 说明文档
```

## 核心类和结构

### MCTSnode

struct MCTSnode {
    // 原有MCTS数据
    int16_t childrennum;
    int16_t legalChildrennum;
    MCTSchild* children;  // 动态分配，根据实际合法走法数量
    uint64_t visits;
    NNUE::ValueSum WRtotal;
    Color nextColor;
    
    // 新增必胜必败判断
    bool isWinDetermined;    // 是否已确定胜负
    bool isWin;              // 是否必胜
    int stepsToWin;          // 到胜负的步数
};
```

### MCTS_CacheTable
```cpp
class MCTS_CacheTable {
public:
    struct Entry {
        Hash128 hash;
        Color maybeWinner;
        int gameEndMovenum;
        NNUE::ValueType nnueValue;
        std::vector<std::pair<Loc, double>> legalMovesWithPolicy;
        Loc bestMove;
    };
    
    MCTS_CacheTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo);
    ~MCTS_CacheTable();
    
    bool get(Hash128 nnHash, Entry& ret);
    void set(const Entry& ent);
    void clear();
};
```

### MCTSsearch

class MCTSsearch {
public:
    static MCTS_CacheTable* cacheTable;  // 全局缓存表
    
    // 缓存管理方法
    static void initializeCache(int sizePowerOfTwo = 20, int mutexPoolSizePowerOfTwo = 12);
    static void clearCache();
    static void destroyCache();
    
    MCTSnode* rootNode;
    NNUEBoardHistory* boardHistory;
    Player attackPlayer;  // VCF搜索的攻击方
    
    // 主要接口
    float fullsearch(Color color, double factor, Loc& bestmove);
    void play(Color color, Loc loc);
    void undo();
    
private:
    // 核心算法
    SearchResult search(MCTSnode* node, uint64_t remainVisits, bool isRoot);
bool checkWinLossDetermined(MCTSnode* node);
int selectChildIDToSearch(MCTSnode* node);
};
```

## 使用方法

### 基本使用
```cpp
// 创建NNUE棋盘历史
NNUEBoardHistory* boardHistory = new NNUEBoardHistory(weights, nnInputParams);
boardHistory->clear(board, C_BLACK, rules);

// 创建新MCTS
MCTSsearch mcts(boardHistory, C_BLACK);  // C_BLACK为攻击方

// 设置参数
mcts.params.puct = 2.0;
mcts.params.expandFactor = 0.2;

// 执行搜索
Loc bestMove;
float value = mcts.fullsearch(C_BLACK, 1.0, bestMove);

// 下棋
mcts.play(C_BLACK, bestMove);
```

### 参数配置
```cpp
struct Param {
    double expandFactor = 0.2;  // 扩展因子
    double puct = 2;            // PUCT常数
    double puctPow = 0.75;      // PUCT幂次
    double puctBase = 10;       // PUCT基数
    double fpuReduction = 0.1;  // FPU减少
    double policyTemp = 1.1;    // 策略温度
};
```

## 编译和测试

### 编译库
```bash
cd mcts
make
```

### 编译并运行测试
```bash
make test
./test_mcts
```

### 清理
```bash
make clean
```

## 算法改进

### 1. 必胜必败检测
- 利用 `getAllVCFAttackOrDefenseLocs` 检测游戏结束状态
- 当 `maybeWinner != C_WALL` 时，游戏结果已确定
- 根据 `gameEndMovenum` 计算到胜负的步数

### 2. 节点状态传播
- 当所有子节点状态确定时，父节点状态也确定
- 如果存在必败子节点，父节点必胜
- 如果所有子节点必胜，父节点必败

### 3. 选择策略优化
- 必胜走法获得最高优先级
- 必败走法获得最低优先级
- 未确定走法按传统PUCT公式选择

### 4. 缓存优化
- 避免重复计算VCF结果
- 避免重复NNUE评估
- 避免重复生成合法走法列表

## 性能特点

1. **搜索效率**：通过必胜必败判断，避免无效搜索
2. **缓存加速**：减少重复计算，提高搜索速度
3. **精确终局**：准确识别游戏结束状态
4. **策略优化**：优先考虑确定性走法

## 与旧版本对比

| 特性 | mcts_old | mcts |
|------|----------|----------|
| 必胜必败判断 | ✗ | ✓ |
| VCF缓存 | ✗ | ✓ |
| NNUE缓存 | 基础 | 完整 |
| 合法走法缓存 | ✗ | ✓ |
| 确定性走法优先 | ✗ | ✓ |
| 搜索剪枝 | 基础 | 增强 |

## 注意事项

1. 需要正确设置 `attackPlayer` 参数
2. 缓存会占用内存，长时间运行需要适当清理
3. VCF计算可能较耗时，缓存效果明显
4. 必胜必败判断依赖于VCF分析的准确性