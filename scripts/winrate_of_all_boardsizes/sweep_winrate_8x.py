import subprocess
import re
from time import sleep
import threading

def run_engine(engine_cmd):
    # 启动引擎进程
    proc = subprocess.Popen(
        engine_cmd.split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    while True:
        line = proc.stderr.readline()
        sleep(0.01)  # 根据引擎响应速度调整等待时间
        #print(line)
        if "beginning main protocol loop" in line:  # 找到目标行
            break
    proc.stdin.write("version\n")
    proc.stdin.flush()
    # 读取stdout直到出现等号
    while "=" not in proc.stdout.readline():
        pass
    #while True:
    #    line = proc.stderr.readline()
    #    if not line:
    #        break
    #    print(line)
    results = []


    unsure_list=[]
    
    # 遍历所有棋盘尺寸组合
    for x in range(2, 9):
        for y in range(2, 9):
            # 发送boardsize命令，确保命令格式正确
            cmd = f"boardsize {x} {y}\n"  # 使用\n而不是\r\n
            proc.stdin.write(cmd)
            proc.stdin.flush()
            #print(f"Sent command: {cmd.strip()}")  # 添加调试信息
            
            # 读取stdout直到出现等号
            while "=" not in proc.stdout.readline():
                pass
            
            # 发送time_settings命令
            proc.stdin.write(f"time_settings 0 5 1\n")
            proc.stdin.flush()
            # 读取stdout直到出现等号
            while "=" not in proc.stdout.readline():
                pass
            proc.stdout.flush()
            
            
            # 发送genmove命令
            proc.stdin.write(f"genmove b\n")
            proc.stdin.flush()
            
            # 等待并获取stderr输出
            sleep(0.5)  # 根据引擎响应速度调整等待时间
            target_line=None
            while True:
                line = proc.stderr.readline()
                sleep(0.1)  # 根据引擎响应速度调整等待时间
                #print(line)
                if not line:
                    break
                if line.startswith(":"):  # 找到目标行
                    print(f"x={x} ,y={y} , line={line}")
                    target_line=line
                    break
            
            proc.stderr.flush()

            while "=" not in proc.stdout.readline():
                pass
            proc.stdout.flush()
            # 提取T值
            t_value = None
            
            if match := re.search(r"W\s+([\d.]+)c", target_line):
                t_value = match.group(1)
            elif match := re.search(r"W\s+-([\d.]+)c", target_line):
                t_value = "-"+match.group(1)
            else:
                print(f"ERROR x={x} ,y={y}")
            
            # 记录结果
            if t_value:
                results.append(f"{x} {y} {t_value}\n")
                print(f"Processed {x}x{y}: {t_value}")
                if(float(t_value)>-98 and float(t_value)<98):
                    unsure_list.append([x,y])
                    print(f"{x}x{y} will recalculate with longer time")
                # 保存结果到文件
                else:
                    with open("results_bt8x8.txt", "a") as f:
                        print(f"{x} {y} {t_value}",file=f)
    
    # 遍历所有棋盘尺寸组合
    for xy in unsure_list:
        x=xy[0]
        y=xy[1]
        # 发送boardsize命令，确保命令格式正确
        cmd = f"boardsize {x} {y}\n"  # 使用\n而不是\r\n
        proc.stdin.write(cmd)
        proc.stdin.flush()
        #print(f"Sent command: {cmd.strip()}")  # 添加调试信息
        
        # 读取stdout直到出现等号
        while "=" not in proc.stdout.readline():
            pass
        
        # 发送time_settings命令
        proc.stdin.write(f"time_settings 0 120 1\n")
        proc.stdin.flush()
        # 读取stdout直到出现等号
        while "=" not in proc.stdout.readline():
            pass
        proc.stdout.flush()
        
        # 发送genmove命令
        proc.stdin.write(f"genmove b\n")
        proc.stdin.flush()
        
        # 等待并获取stderr输出
        sleep(0.5)  # 根据引擎响应速度调整等待时间
        target_line=None
        while True:
            line = proc.stderr.readline()
            sleep(0.1)  # 根据引擎响应速度调整等待时间
            #print(line)
            if not line:
                break
            if line.startswith(":"):  # 找到目标行
                print(f"x={x} ,y={y} , line={line}")
                target_line=line
                break
        proc.stderr.flush()
        while "=" not in proc.stdout.readline():
            pass
        proc.stdout.flush()
        # 提取T值
        t_value = None
        
        if match := re.search(r"W\s+([\d.]+)c", target_line):
            t_value = match.group(1)
        elif match := re.search(r"W\s+-([\d.]+)c", target_line):
            t_value = "-"+match.group(1)
        else:
            print(f"ERROR x={x} ,y={y}")
        
        # 记录结果
        if t_value:
            results.append(f"{x} {y} {t_value}\n")
            print(f"Processed {x}x{y}: {t_value}")
            with open("results_bt8x8.txt", "a") as f:
                print(f"{x} {y} {t_value}",file=f)
                 

    proc.terminate()

if __name__ == "__main__":
    # 示例引擎启动命令（需要替换为实际命令）
    engine_command = "D:/katago/engine2025/breakthrough8x.exe gtp -config ./engine2024.cfg -model D:/kata2025/breakthrough8x/data/latest.bin.gz"
    run_engine(engine_command)