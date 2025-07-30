import random

def generate_hash128():
    # 生成两个64位无符号整数，符合C++中Hash128的构造方式
    high = random.getrandbits(64)
    low = random.getrandbits(64)
    # 格式化为C++代码格式，使用ULL后缀表示无符号长long
    return f"Hash128(0x{high:016x}ULL, 0x{low:016x}ULL)"

# 为3种循环规则生成哈希值（SEVENTHREE, NONE, REPEATEND）
if __name__ == "__main__":
    num_hashes = 3  # 与loopRule规则数量匹配
    print("const Hash128 Rules::ZOBRIST_LOOPRULE_RULE_HASH[3] = {")
    for i in range(num_hashes):
        hash_str = generate_hash128()
        # 最后一个元素不加逗号
        print(f"  {hash_str}{',' if i < num_hashes - 1 else ''}")
    print("};")