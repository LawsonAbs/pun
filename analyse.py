"""
分析结果文件中有哪些错误样例
"""

def analyse(path):
    cur = ""
    error_flag = 0 # 错误预测的标志
    cnt = 0 # 计算最后预测失败的
    with open(path,"r") as f:
        line = f.readline()        
        while line:             
            if line == '\n': # 如果是单纯的换行
                if error_flag:
                    print(cur)
                    error_flag = 0
                cur = "" # 重置
                line = f.readline()
                continue
            else :
                line = line.strip() # 去换行
                line = line.split() # 生成数组                
                cur += (line[0] + " ")                
                if (line[1] == 'P' and line[2]!='P') or (line[1] != 'P' and line[2]=='P'): # 如果两者的结果不匹配
                    error_flag = 1 # 错误样例
                    cnt += 1 
            line = f.readline()
    return cnt

if __name__ == "__main__":
    base = "/home/lawson/program/punLocation/scores/homo_pron/all_"
    error = 0
    for i in range(10):
        path = base+ str(i)        
        num = analyse(path)        
        error += num
        print(f"========={i}. => {num}===============================")
    print(f"{error}")