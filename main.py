import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import datetime
import sys
import akshare as ak

class ChanStrategy:
    def __init__(self, symbol, df):
        self.symbol = symbol
        self.df = df.copy()
        # 1. 计算指标 (修复变量引用)
        self.prepare_indicators()
        self.k_data = pd.DataFrame()
        self.bi = []
        self.zhongshu = None
        self.buy_point = None
        # 2. 缠论处理
        self.process_chan()

    def prepare_indicators(self):
        """预计算MACD用于背驰判断"""
        ema12 = self.df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['dif'] = ema12 - ema26
        self.df['dea'] = self.df['dif'].ewm(span=9, adjust=False).mean()
        self.df['macd'] = (self.df['dif'] - self.df['dea']) * 2
        self.df['macd_area'] = self.df['macd'].abs()

    def clean_inclusion(self):
        """严格K线包含处理"""
        if len(self.df) < 2: return
        k_list = []
        last_k = {'time': self.df.index[0], 'high': self.df.iloc[0]['High'], 'low': self.df.iloc[0]['Low'], 'idx': 0}
        direction = 1 
        for i in range(1, len(self.df)):
            curr_h, curr_l = self.df.iloc[i]['High'], self.df.iloc[i]['Low']
            if (last_k['high'] >= curr_h and last_k['low'] <= curr_l) or (curr_h >= last_k['high'] and curr_l <= last_k['low']):
                if direction == 1:
                    last_k['high'], last_k['low'] = max(last_k['high'], curr_h), max(last_k['low'], curr_l)
                else:
                    last_k['high'], last_k['low'] = min(last_k['high'], curr_h), min(last_k['low'], curr_l)
            else:
                direction = 1 if curr_h > last_k['high'] else -1
                k_list.append(last_k)
                last_k = {'time': self.df.index[i], 'high': curr_h, 'low': curr_l, 'idx': i}
        k_list.append(last_k)
        self.k_data = pd.DataFrame(k_list)

    def identify_bi(self):
        """标准化笔识别"""
        k = self.k_data
        if len(k) < 5: return
        nodes = []
        for i in range(1, len(k) - 1):
            if k.iloc[i]['high'] > k.iloc[i-1]['high'] and k.iloc[i]['high'] > k.iloc[i+1]['high']:
                nodes.append({'type': 'top', 'val': k.iloc[i]['high'], 'idx': k.iloc[i]['idx'], 'time': k.iloc[i]['time']})
            elif k.iloc[i]['low'] < k.iloc[i-1]['low'] and k.iloc[i]['low'] < k.iloc[i+1]['low']:
                nodes.append({'type': 'bottom', 'val': k.iloc[i]['low'], 'idx': k.iloc[i]['idx'], 'time': k.iloc[i]['time']})
        
        bi = []
        for n in nodes:
            if not bi: bi.append(n)
            elif n['type'] != bi[-1]['type']:
                if abs(n['idx'] - bi[-1]['idx']) >= 3: bi.append(n) # 间距要求
            else:
                if (n['type'] == 'top' and n['val'] > bi[-1]['val']) or (n['type'] == 'bottom' and n['val'] < bi[-1]['val']):
                    bi[-1] = n
        self.bi = bi

    def get_macd_power(self, start_time, end_time):
        return self.df.loc[start_time:end_time, 'macd_area'].sum()

    def analyze_three_buy(self):
        """判定逻辑"""
        if len(self.bi) < 5: return # 降低要求，至少5笔
        
        try:
            # 自动适应笔数，取最后的中枢
            m1, m2, m3 = self.bi[-5], self.bi[-4], self.bi[-3]
            zg = min(max(m1['val'], m2['val']), max(m2['val'], m3['val']))
            zd = max(min(m1['val'], m2['val']), min(m2['val'], m3['val']))
            
            if zd >= zg: return
            self.zhongshu = {'zg': zg, 'zd': zd, 'start': m1['time'], 'end': m3['time']}

            b_leave = self.bi[-2] 
            b_back = self.bi[-1]

            # 1. 价格过滤
            if b_back['val'] <= zg: return # 跌回中枢，pass
            
            # 2. 力度过滤 (适度放宽到 0.8)
            p_leave = self.get_macd_power(self.bi[-3]['time'], self.bi[-2]['time'])
            p_back = self.get_macd_power(self.bi[-2]['time'], self.bi[-1]['time'])
            if p_back > p_leave * 0.8: return 

            # 3. 确认回调笔已企稳（当前收盘价不低于回调底）
            if self.df.iloc[-1]['Close'] < b_back['val']: return
            
            self.buy_point = b_back
        except: pass

    def process_chan(self):
        self.clean_inclusion()
        self.identify_bi()
        self.analyze_three_buy()

    def generate_plot_html(self):
        if not self.buy_point: return ""
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
        fig.add_trace(go.Candlestick(x=self.df.index, open=self.df['Open'], high=self.df['High'], low=self.df['Low'], close=self.df['Close'], name='K线'), row=1, col=1)
        bi_x = [b['time'] for b in self.bi]; bi_y = [b['val'] for b in self.bi]
        fig.add_trace(go.Scatter(x=bi_x, y=bi_y, mode='lines+markers', name='笔', line=dict(color='orange')), row=1, col=1)
        if self.zhongshu:
            fig.add_shape(type="rect", x0=self.zhongshu['start'], y0=self.zhongshu['zd'], x1=self.df.index[-1], y1=self.zhongshu['zg'], fillcolor="rgba(255,0,0,0.1)", line_width=0, row=1, col=1)
        fig.add_trace(go.Bar(x=self.df.index, y=self.df['macd'], name='MACD'), row=2, col=1)
        fig.update_layout(title=f"{self.symbol} 三买选股报告", xaxis_rangeslider_visible=False, height=700)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

# --- 指数获取函数 (保持最新修复版) ---
def get_tickers(index_name):
    print(f"正在获取 {index_name} 列表...")
    try:
        if index_name == "HSI":
            df = ak.stock_hk_index_stock_cons_em(symbol="恒生指数")
            tickers = [f"{c[1:] if len(c)==5 else c}.HK" for c in df['代码'].tolist()]
        elif index_name == "HS300":
            df = ak.index_stock_cons(symbol="000300")
            tickers = [f"{c}.SS" if c.startswith('6') else f"{c}.SZ" for c in df['品种代码'].tolist()]
        elif index_name == "ZZ500":
            df = ak.index_stock_cons(symbol="000905")
            tickers = [f"{c}.SS" if c.startswith('6') else f"{c}.SZ" for c in df['品种代码'].tolist()]
        elif index_name == "NDX":
            # 简化版NDX
            tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOG", "AMZN", "META", "AMD", "AVGO", "COST"]
        else: tickers = ["0700.HK"]
        return sorted(list(set(tickers)))
    except: return ["0700.HK"]

def process_stock(symbol):
    try:
        # 将 period 从 2mo 改为 59d，确保在 Yahoo 的 60 天限制内
        data = yf.download(symbol, period="59d", interval="30m", progress=False, timeout=10)
        
        if data.empty or len(data) < 50: 
            return None
            
        # 移除多级索引（针对新版 yfinance 兼容）
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data.index = data.index.tz_localize(None)
        
        cs = ChanStrategy(symbol, data)
        if cs.buy_point:
            print(f"找到信号: {symbol}")
            return cs.generate_plot_html()
    except Exception as e:
        # print(f"下载 {symbol} 出错: {e}")
        pass
    return None

def main():
    index_arg = sys.argv[1] if len(sys.argv) > 1 else "HSI"
    tickers = get_tickers(index_arg)
    print(f"去重后共 {len(tickers)} 只股票，开始扫描...")
    
    charts = []
    # 增加 max_workers 提高速度
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_stock, s): s for s in tickers}
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: charts.append(res)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(f"<html><head><meta charset='utf-8'></head><body><h1>{index_arg} 扫描报告</h1>")
        if not charts: f.write("<h2>今日未发现符合强力三买的标的</h2>")
        else:
            for c in charts: f.write(f"<div>{c}</div><hr>")
        f.write("</body></html>")

if __name__ == "__main__":
    main()
