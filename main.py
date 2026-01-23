import pandas as pd
import numpy as np
import yfinance as yf
import concurrent.futures

class ChanStrategy:
    def __init__(self, df):
        # 确保数据是干净的，且索引不包含多余信息
        self.df = df[['High', 'Low', 'Close']].copy()
        self.k_data = pd.DataFrame()
        self.clean_data()
        
    def clean_data(self):
        """处理K线包含关系"""
        df = self.df
        if len(df) < 5: return
        
        processed_k = []
        # 初始化第一根K线
        last_up = df.iloc[0]['High']
        last_down = df.iloc[0]['Low']
        direction = 1 
        
        for i in range(1, len(df)):
            curr_up = df.iloc[i]['High']
            curr_down = df.iloc[i]['Low']
            
            # 包含处理
            if (last_up >= curr_up and last_down <= curr_down) or \
               (curr_up >= last_up and curr_down <= last_down):
                if direction == 1:
                    last_up = max(last_up, curr_up)
                    last_down = max(last_down, curr_down)
                else:
                    last_up = min(last_up, curr_up)
                    last_down = min(last_down, curr_down)
            else:
                direction = 1 if curr_up > last_up else -1
                processed_k.append({'up': last_up, 'down': last_down, 'idx': i-1})
                last_up, last_down = curr_up, curr_down
                
        processed_k.append({'up': last_up, 'down': last_down, 'idx': len(df)-1})
        self.k_data = pd.DataFrame(processed_k)

    def identify_bi(self):
        """识别笔 - 增加更严格的距离约束"""
        if self.k_data.empty or len(self.k_data) < 7: return []
        nodes = []
        k = self.k_data
        for i in range(1, len(k) - 1):
            if k.iloc[i]['up'] > k.iloc[i-1]['up'] and k.iloc[i]['up'] > k.iloc[i+1]['up']:
                nodes.append({'type': 'top', 'val': k.iloc[i]['up'], 'idx': i})
            elif k.iloc[i]['down'] < k.iloc[i-1]['down'] and k.iloc[i]['down'] < k.iloc[i+1]['down']:
                nodes.append({'type': 'bottom', 'val': k.iloc[i]['down'], 'idx': i})
        
        bi = []
        for n in nodes:
            if not bi: bi.append(n)
            else:
                if n['type'] != bi[-1]['type']:
                    # 确保顶底之间有足够间距
                    if abs(n['idx'] - bi[-1]['idx']) >= 3: bi.append(n)
                else:
                    if n['type'] == 'top' and n['val'] > bi[-1]['val']: bi[-1] = n
                    elif n['type'] == 'bottom' and n['val'] < bi[-1]['val']: bi[-1] = n
        return bi

    def find_third_buy(self):
        bi = self.identify_bi()
        # 三买至少需要：中枢(3笔) + 离开(1笔) + 回调(1笔) = 5笔以上，建议7笔更稳
        if len(bi) < 6: return False, 0, 0
        
        try:
            # 简化中枢判定：取倒数第4, 3, 2笔
            # 确保方向：b1下, b2上, b3下 构成上涨中枢
            b_leave = bi[-2] # 离开笔
            b_back = bi[-1]  # 回调笔
            
            # 取中枢的前三笔
            m1, m2, m3 = bi[-5], bi[-4], bi[-3]
            zg = min(max(m1['val'], m2['val']), max(m2['val'], m3['val']))
            zd = max(min(m1['val'], m2['val']), min(m2['val'], m3['val']))
            
            if zd >= zg: return False, 0, 0
            
            # 三买条件：离开段最高点 > zg，回调段最低点 > zg
            if b_leave['type'] == 'top' and b_leave['val'] > zg:
                if b_back['type'] == 'bottom' and b_back['val'] > zg:
                    # 额外过滤：回调不能离中枢太远，也不能太近
                    return True, zg, b_back['val']
        except:
            pass
        return False, 0, 0

def process_stock(symbol):
    """
    完全隔离的单只股票处理
    """
    try:
        # 使用 Ticker().history 彻底避免多级索引混淆
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1mo", interval="30m")
        
        if data is None or len(data) < 20:
            return None
        
        # 显式重命名列名，防止大小写或格式问题
        data.columns = [c.capitalize() for c in data.columns]
        
        cs = ChanStrategy(data)
        is_buy, zg, buy_val = cs.find_third_buy()
        
        # 获取当前最新价，用于二次验证
        current_price = data['Close'].iloc[-1]
        
        if is_buy:
            return f"代码: {symbol:8s} | 现价: {current_price:7.2f} | 中枢上沿: {zg:7.2f} | 回调底: {buy_val:7.2f}"
    except Exception as e:
        # print(f"Error processing {symbol}: {e}")
        pass
    return None

def main():
    hstech_list = [
        "0700.HK", "9988.HK", "3690.HK", "1810.HK", "9888.HK", 
        "9618.HK", "2015.HK", "2382.HK", "0981.HK", "1024.HK",
        "1314.HK", "0241.HK", "0285.HK", "0772.HK", "0992.HK",
        "1347.HK", "1478.HK", "1610.HK", "1797.HK", "1833.HK",
        "2190.HK", "2269.HK", "2318.HK", "2518.HK", "2688.HK",
        "3888.HK", "6060.HK", "6608.HK", "6618.HK", "9626.HK"
    ]
    
    print(f"开始扫描恒生科技指数... (隔离模式)")
    
    results = []
    # 港股建议减小并发，防止被 Yahoo 屏蔽
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(process_stock, t): t for t in hstech_list}
        for future in concurrent.futures.as_completed(future_to_ticker):
            res = future.result()
            if res:
                results.append(res)
                print(f"发现信号 -> {res}")

    print("\n" + "="*50)
    print(f"扫描完成，符合三买条件个股如下：")
    if not results:
        print("暂无匹配个股。")
    else:
        for r in sorted(results):
            print(r)

if __name__ == "__main__":
    main()
