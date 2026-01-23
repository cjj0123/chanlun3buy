import pandas as pd
import numpy as np
import yfinance as yf

class ChanStrategy:
    def __init__(self, df):
        self.df = df.copy()
        self.clean_data()
        
    def clean_data(self):
        """步骤1 & 2：处理K线包含关系"""
        df = self.df
        df['up'] = df['High']
        df['down'] = df['Low']
        
        # 包含处理逻辑
        processed_k = []
        if len(df) > 0:
            last_k = df.iloc[0].to_dict()
            direction = 1 # 初始方向
            
            for i in range(1, len(df)):
                curr_k = df.iloc[i].to_dict()
                # 判断包含关系
                is_contained = (last_k['up'] >= curr_k['up'] and last_k['down'] <= curr_k['down']) or \
                               (curr_k['up'] >= last_k['up'] and curr_k['down'] <= last_k['down'])
                
                if is_contained:
                    # 向上趋势：取高高，向下趋势：取低低
                    if direction == 1:
                        last_k['up'] = max(last_k['up'], curr_k['up'])
                        last_k['down'] = max(last_k['down'], curr_k['down'])
                    else:
                        last_k['up'] = min(last_k['up'], curr_k['up'])
                        last_k['down'] = min(last_k['down'], curr_k['down'])
                else:
                    direction = 1 if curr_k['up'] > last_k['up'] else -1
                    processed_k.append(last_k)
                    last_k = curr_k
            processed_k.append(last_k)
        self.k_data = pd.DataFrame(processed_k)

    def identify_bi(self):
        """步骤2简化：识别笔 (Bi)"""
        # 寻找顶底分型
        nodes = []
        k = self.k_data
        for i in range(1, len(k) - 1):
            # 顶分型
            if k.iloc[i]['up'] > k.iloc[i-1]['up'] and k.iloc[i]['up'] > k.iloc[i+1]['up']:
                nodes.append({'type': 'top', 'val': k.iloc[i]['up'], 'idx': i})
            # 底分型
            elif k.iloc[i]['down'] < k.iloc[i-1]['down'] and k.iloc[i]['down'] < k.iloc[i+1]['down']:
                nodes.append({'type': 'bottom', 'val': k.iloc[i]['down'], 'idx': i})
        
        # 过滤相邻的相同类型和距离太近的点（确保笔的成立）
        bi = []
        for n in nodes:
            if not bi:
                bi.append(n)
            else:
                if n['type'] != bi[-1]['type']:
                    if abs(n['idx'] - bi[-1]['idx']) >= 3: # 笔的距离阈值
                        bi.append(n)
                else:
                    # 同类型取极值
                    if n['type'] == 'top' and n['val'] > bi[-1]['val']:
                        bi[-1] = n
                    elif n['type'] == 'bottom' and n['val'] < bi[-1]['val']:
                        bi[-1] = n
        return bi

    def find_third_buy(self):
        """步骤4, 5 & 6：中枢识别与三买判断"""
        bi = self.identify_bi()
        if len(bi) < 7: return False # 至少需要足够笔数构成中枢+离开+回调
        
        # 1. 寻找最近的一个中枢 (由前3笔重叠部分构成)
        # 假设 bi[-5], bi[-4], bi[-3] 构成中枢
        # 中枢区间 [ZD, ZG]
        b1, b2, b3 = bi[-7], bi[-6], bi[-5]
        
        # 中枢高点取Min(b1_high, b2_high, b3_high), 低点取Max(b1_low...)
        # 简化版逻辑：
        zg = min(max(b1['val'], bi[-8]['val'] if len(bi)>8 else b1['val']), 
                 max(b2['val'], b3['val']))
        zd = max(min(b1['val'], bi[-8]['val'] if len(bi)>8 else b1['val']), 
                 min(b2['val'], b3['val']))
        
        if zd >= zg: return False # 不构成重叠中枢

        # 2. 离开段：bi[-2] 向上离开
        leave_segment = bi[-2]
        if leave_segment['type'] != 'top' or leave_segment['val'] <= zg:
            return False

        # 3. 回调段：bi[-1] 回调且不破中枢上沿 ZG
        back_segment = bi[-1]
        current_price = self.df.iloc[-1]['Close']
        
        # 条件：回调段是向下的，且最低点 > 中枢上沿
        if back_segment['type'] == 'bottom':
            if back_segment['val'] > zg:
                # 辅助指标过滤：MACD (可选)
                # if macd_filter(): ...
                return True, zg, back_segment['val']
        
        return False

def scan_stocks(symbol_list):
    """批量选股器"""
    results = []
    print(f"开始扫描股票，共 {len(symbol_list)} 只...")
    
    for symbol in symbol_list:
        try:
            # 获取30分钟数据 (yfinance支持)
            data = yf.download(symbol, period="5d", interval="30m", progress=False)
            if len(data) < 50: continue
            
            cs = ChanStrategy(data)
            is_buy, zg, buy_val = cs.find_third_buy()
            
            if is_buy:
                print(f"找到符合三买个股: {symbol} | 中枢上沿: {zg:.2f} | 当前回调底: {buy_val:.2f}")
                results.append(symbol)
        except Exception as e:
            print(f"分析 {symbol} 出错: {e}")
            
    return results

# --- 测试运行 ---
if __name__ == "__main__":
    # 示例股票池（美股或港股/A股需对应格式）
    test_list = ["AAPL", "TSLA", "NVDA", "MSFT", "AMD", "GOOGL"]
    
    selected_stocks = scan_stocks(test_list)
    print("\n最终选股结果:", selected_stocks)
