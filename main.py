import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
import concurrent.futures
import time

class ChanStrategy:
    def __init__(self, df):
        if df is None or len(df) < 10:
            self.k_data = pd.DataFrame()
            return
        
        # 解决 yfinance 多级索引问题
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        self.df = df.copy()
        self.clean_data()
        
    def clean_data(self):
        """处理K线包含关系"""
        df = self.df
        # 确保列名正确
        high_col = 'High'
        low_col = 'Low'
        
        processed_k = []
        if len(df) > 0:
            # 初始化第一根K线
            last_up = df.iloc[0][high_col]
            last_down = df.iloc[0][low_col]
            direction = 1 
            
            for i in range(1, len(df)):
                curr_up = df.iloc[i][high_col]
                curr_down = df.iloc[i][low_col]
                
                # 判断包含关系
                is_contained = (last_up >= curr_up and last_down <= curr_down) or \
                               (curr_up >= last_up and curr_down <= last_down)
                
                if is_contained:
                    if direction == 1: # 向上趋势
                        last_up = max(last_up, curr_up)
                        last_down = max(last_down, curr_down)
                    else: # 向下趋势
                        last_up = min(last_up, curr_up)
                        last_down = min(last_down, curr_down)
                else:
                    direction = 1 if curr_up > last_up else -1
                    processed_k.append({'up': last_up, 'down': last_down, 'idx': i-1})
                    last_up, last_down = curr_up, curr_down
            processed_k.append({'up': last_up, 'down': last_down, 'idx': len(df)-1})
        self.k_data = pd.DataFrame(processed_k)

    def identify_bi(self):
        """识别笔"""
        if self.k_data.empty or len(self.k_data) < 5: return []
        nodes = []
        k = self.k_data
        for i in range(1, len(k) - 1):
            if k.iloc[i]['up'] > k.iloc[i-1]['up'] and k.iloc[i]['up'] > k.iloc[i+1]['up']:
                nodes.append({'type': 'top', 'val': k.iloc[i]['up'], 'idx': k.iloc[i]['idx']})
            elif k.iloc[i]['down'] < k.iloc[i-1]['down'] and k.iloc[i]['down'] < k.iloc[i+1]['down']:
                nodes.append({'type': 'bottom', 'val': k.iloc[i]['down'], 'idx': k.iloc[i]['idx']})
        
        bi = []
        for n in nodes:
            if not bi: bi.append(n)
            else:
                if n['type'] != bi[-1]['type']:
                    if abs(n['idx'] - bi[-1]['idx']) >= 3: bi.append(n)
                else:
                    if n['type'] == 'top' and n['val'] > bi[-1]['val']: bi[-1] = n
                    elif n['type'] == 'bottom' and n['val'] < bi[-1]['val']: bi[-1] = n
        return bi

    def find_third_buy(self):
        """三买判断核心逻辑"""
        bi = self.identify_bi()
        if len(bi) < 5: return False, 0, 0
        
        # 简单逻辑：寻找最近一个中枢
        # 取倒数第5,4,3笔构成中枢
        try:
            b1, b2, b3 = bi[-5], bi[-4], bi[-3]
            zg = min(max(b1['val'], bi[-6]['val'] if len(bi)>5 else b1['val']), max(b2['val'], b3['val']))
            zd = max(min(b1['val'], bi[-6]['val'] if len(bi)>5 else b1['val']), min(b2['val'], b3['val']))
            
            if zd >= zg: return False, 0, 0
            
            leave_segment = bi[-2] # 离开段
            back_segment = bi[-1]  # 回调段
            
            if leave_segment['type'] == 'top' and leave_segment['val'] > zg:
                if back_segment['type'] == 'bottom' and back_segment['val'] > zg:
                    return True, zg, back_segment['val']
        except:
            pass
        return False, 0, 0

def get_all_tickers():
    """获取全市场股票代码列表"""
    tickers = []
    print("正在获取全市场股票列表...")
    try:
        # 1. A股 (使用akshare) - 取上证和深证前500只活跃股（为了演示速度，你可以去掉[:500]）
        df_a = ak.stock_zh_a_spot_em()
        tickers += [f"{code}.SS" if code.startswith('6') else f"{code}.SZ" for code in df_a['代码'].tolist()[:300]]
        
        # 2. 港股 - 取前100只
        df_hk = ak.stock_hk_spot_em()
        tickers += [f"{code[1:]}.HK" for code in df_hk['代码'].tolist()[:100]]
        
        # 3. 美股 - 常用科技股名单（yfinance抓取全美股极慢，建议手动维护重点名单）
        tickers += ["AAPL", "TSLA", "NVDA", "MSFT", "AMD", "GOOG", "AMZN", "META", "NFLX", "BABA"]
    except Exception as e:
        print(f"获取代码列表失败: {e}")
        tickers = ["AAPL", "TSLA", "600519.SS", "000001.SZ"] # 兜底名单
    return list(set(tickers))

def process_stock(symbol):
    """单个股票处理函数"""
    try:
        # 下载30分钟K线 (5天数据)
        data = yf.download(symbol, period="5d", interval="30m", progress=False, timeout=10)
        if data.empty: return None
        
        cs = ChanStrategy(data)
        is_buy, zg, buy_val = cs.find_third_buy()
        
        if is_buy:
            return f"{symbol} | 中枢上沿: {zg:.2f} | 回调底: {buy_val:.2f}"
    except:
        pass
    return None

def main():
    all_tickers = get_all_tickers()
    print(f"开始扫描，共计 {len(all_tickers)} 只股票...")
    
    results = []
    # 使用线程池并发加速
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(process_stock, ticker): ticker for ticker in all_tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            res = future.result()
            if res:
                print(f"找到信号: {res}")
                results.append(res)
    
    print("\n--- 最终选股结果 ---")
    if not results:
        print("今日无三买信号")
    else:
        for r in results:
            print(r)
    
    # 将结果写入文件，方便GitHub Actions展示
    with open("results.txt", "w") as f:
        f.write("\n".join(results) if results else "No signals found.")

if __name__ == "__main__":
    main()
