import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
import concurrent.futures
import time

# --- 缠论逻辑类 (保持不变) ---
class ChanStrategy:
    def __init__(self, df):
        if df is None or len(df) < 20: # 30分钟线需要更多数据来构建中枢
            self.k_data = pd.DataFrame()
            return
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        self.df = df.copy()
        self.clean_data()
        
    def clean_data(self):
        df = self.df
        high_col, low_col = 'High', 'Low'
        processed_k = []
        if len(df) > 0:
            last_up, last_down = df.iloc[0][high_col], df.iloc[0][low_col]
            direction = 1 
            for i in range(1, len(df)):
                curr_up, curr_down = df.iloc[i][high_col], df.iloc[i][low_col]
                is_contained = (last_up >= curr_up and last_down <= curr_down) or \
                               (curr_up >= last_up and curr_down <= last_down)
                if is_contained:
                    if direction == 1:
                        last_up, last_down = max(last_up, curr_up), max(last_down, curr_down)
                    else:
                        last_up, last_down = min(last_up, curr_up), min(last_down, curr_down)
                else:
                    direction = 1 if curr_up > last_up else -1
                    processed_k.append({'up': last_up, 'down': last_down, 'idx': i-1})
                    last_up, last_down = curr_up, curr_down
            processed_k.append({'up': last_up, 'down': last_down, 'idx': len(df)-1})
        self.k_data = pd.DataFrame(processed_k)

    def identify_bi(self):
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
        bi = self.identify_bi()
        if len(bi) < 7: return False, 0, 0
        try:
            # 取最近的一个中枢结构
            b1, b2, b3 = bi[-7], bi[-6], bi[-5]
            zg = min(max(b1['val'], bi[-8]['val'] if len(bi)>7 else b1['val']), max(b2['val'], b3['val']))
            zd = max(min(b1['val'], bi[-8]['val'] if len(bi)>7 else b1['val']), min(b2['val'], b3['val']))
            if zd >= zg: return False, 0, 0
            
            leave_segment = bi[-2] # 向上离开段
            back_segment = bi[-1]  # 向下回调段
            
            if leave_segment['type'] == 'top' and leave_segment['val'] > zg:
                # 核心：回调底不破中枢高点 zg
                if back_segment['type'] == 'bottom' and back_segment['val'] > zg:
                    return True, zg, back_segment['val']
        except: pass
        return False, 0, 0

# --- 重点修改部分：获取恒生科技成分股 ---
def get_hstech_tickers():
    """获取恒生科技指数成分股"""
    print("正在获取恒生科技指数成分股列表...")
    try:
        # 使用 akshare 获取恒生指数成分股数据
        # 恒生科技指数代码通常为 HSTECH
        df = ak.stock_hk_index_spot_em()
        # 过滤出恒生科技指数（这里根据名称过滤更直观）
        # 或者直接使用硬编码的 30 只，因为它们非常固定且重要
        hstech_list = [
            "00700", "09988", "03690", "01810", "09888", 
            "09618", "02015", "02382", "00981", "01024",
            "01314", "0241", "00285", "00772", "00992",
            "01347", "01478", "01610", "01797", "01833",
            "02190", "02269", "02318", "02518", "02688",
            "03888", "06060", "06608", "06618", "09626"
        ]
        # 转换为 yfinance 格式 (例如 00700 -> 0700.HK)
        formatted_tickers = [f"{code[1:] if len(code)==5 else code}.HK" for code in hstech_list]
        return formatted_tickers
    except Exception as e:
        print(f"获取列表失败，使用备份名单: {e}")
        return ["0700.HK", "9988.HK", "3690.HK", "1810.HK", "9888.HK", "9618.HK", "2015.HK"]

def process_stock(symbol):
    try:
        # 获取最近 1 个月数据以确保中枢完整性
        data = yf.download(symbol, period="1mo", interval="30m", progress=False, timeout=15)
        if data.empty: return None
        
        cs = ChanStrategy(data)
        is_buy, zg, buy_val = cs.find_third_buy()
        
        if is_buy:
            return f"【三买确认】 {symbol} | 中枢上沿: {zg:.2f} | 回调底: {buy_val:.2f}"
    except:
        pass
    return None

def main():
    tickers = get_hstech_tickers()
    print(f"开始扫描恒生科技指数，共计 {len(tickers)} 只...")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(process_stock, t): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            res = future.result()
            if res:
                print(res)
                results.append(res)
    
    print("\n" + "="*30)
    print("扫描完成！")
    if not results:
        print("今日恒生科技成分股中未发现 30分钟 三买信号。")
    else:
        print(f"发现以下 {len(results)} 只个股符合三买：")
        for r in results:
            print(r)
    
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results) if results else "No HSTECH signals today.")

if __name__ == "__main__":
    main()
